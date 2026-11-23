"""Historical batch validation, identity checks, and isolated PostgreSQL acceptance."""
from collections import Counter
import json
import os
from pathlib import Path
import unittest

import pandas as pd

from fb_app_historical_clv import FIELDS, compact, digest, overlay
from import_historical_clv import batch, script


def sample():
    raw = ["B-fake", "2026-11-23 12:34", "2026-11-23", "主隊", "客隊", None,
           "入球大小", "Under", 2.5, 2.22]
    row = dict(zip(FIELDS, raw, strict=True), Status="Open", Unit_Profit=0.0, Result_Label="", System_Stake=0.0)
    e = dict(row, bet_id=raw[0], original_json=compact(raw), closing_odds=1.95)
    e["fingerprint"] = digest(e["original_json"])
    return row, e


def classify(group="historical_unknown", clv=None):
    return pd.DataFrame([{"group": group, "clv": clv, "closing": None, "close_time": None, "reason": None}],
                        index=[25]).astype({"clv": "float64"})


class Overlay(unittest.TestCase):
    def test_pending_settled_zero_and_input_not_modified(self):
        row, e = sample()
        ledger = pd.DataFrame([row], index=[25])
        native = classify()
        out, official, shown = overlay(ledger, native, [e])
        self.assertEqual(out.at[25, "Status"], "Open")
        self.assertEqual(official.at[25, "group"], "historical_verified")
        self.assertAlmostEqual(shown[0]["CLV%"], 13.846154)
        e.update(closing_odds=2.22, Status="Settled", Unit_Profit=1.22)
        out, official, shown = overlay(ledger, native, [e])
        self.assertEqual((out.at[25, "Status"], out.at[25, "Unit_Profit"]), ("Settled", 1.22))
        self.assertEqual(official.at[25, "clv"], 0)
        self.assertEqual(shown[0]["狀態"], "已結算")
        self.assertEqual((ledger.at[25, "Status"], native.at[25, "group"]), ("Open", "historical_unknown"))

    def test_unsafe_or_absent_matches_do_not_enter_stats(self):
        row, e = sample()
        df = pd.DataFrame([row], index=[25])
        for change in ({"ID": None}, {"Status": None}, {"Initial_Odds": 2.21}, {"Initial_Line": 3.0},
                       {"Tipsme_Match_ID": "116367"}, {"fingerprint": "f"*64}, {"closing_odds": "NaN"},
                       {"Date": "2026-11-23 12:34:54"}, {"Selection": "Over"}):
            with self.subTest(change=change):
                _, g, visible = overlay(df, classify(), [dict(e, **change)])
                self.assertEqual(g.at[25, "group"], "historical_unknown")
                self.assertFalse(visible)
        self.assertFalse(overlay(df, classify(), [e, e])[2])
        self.assertFalse(overlay(df, classify("follow"), [e])[2])
        self.assertFalse(overlay(df.assign(Home_Team="不同"), classify(), [e])[2])

    def test_existing_native_snapshot_wins_once(self):
        row, e = sample()
        out, g, shown = overlay(pd.DataFrame([row], index=[25]), classify("snapshot", 0.2), [e])
        self.assertEqual((g.at[25, "group"], g.at[25, "clv"]), ("snapshot", .2))
        self.assertEqual(shown[0]["主統計核證來源"], "賽前快照")


@unittest.skipUnless(os.getenv("FB_CLV_REPLAY_DSN") and os.getenv("FB_CLV_FORMAL") and os.getenv("FB_CLV_FULL"),
                     "requires a disposable local PostgreSQL socket and the original exports")
class PostgreSQL(unittest.TestCase):
    def test_full_batch_is_atomic_idempotent_and_private(self):
        import psycopg
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
        from fb_app_historical_clv import sql_rows

        dsn = os.environ["FB_CLV_REPLAY_DSN"]
        info = psycopg.conninfo.conninfo_to_dict(dsn)
        self.assertTrue(info["host"].startswith("/tmp/fb-clv-pg."))
        self.assertEqual(info["port"], "5449")
        rows = list(batch(os.environ["FB_CLV_FORMAL"], os.environ["FB_CLV_FULL"]))
        self.assertEqual(Counter(r["provenance"]["entry_method"] for r in rows), {"A1": 241, "A2": 467})
        with psycopg.connect(dsn) as c, c.cursor() as cur:
            cur.execute("""CREATE TABLE public.football_bets ("ID" text, "Date" text, "Match_Date" text,
                "Home_Team" text, "Away_Team" text,"Tipsme_Match_ID" text, "Bet_Type" text,
                "Selection" text, "Initial_Line" float8, "Initial_Odds" float8, "Status" text, "Note" text,
                "Odds_History" text, "Unit_Profit" float8, "Result_Label" text, "System_Stake" float8);
                CREATE UNIQUE INDEX ux_football_bets_id ON public.football_bets ("ID");
                CREATE TABLE public.football_bets_deleted (id text PRIMARY KEY);""")
            cur.executemany('INSERT INTO public.football_bets VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,'
                            '\'[]\',0,\'\',0)',
                            [(*json.loads(r["raw"]), r["provenance"]["status_at_export"]) for r in rows])
            before = cur.execute("SELECT md5(string_agg(row_to_json(b)::text,'' ORDER BY \"ID\")) "
                                 "FROM public.football_bets b").fetchone()[0]
            self.assertEqual(len(rows), 708)
            planned = cur.execute(script(rows, with_table=False)).fetchone()
            self.assertEqual([r["n"] for r in planned[1]], [27, 681])
            self.assertEqual(planned[3][0], {"status": "Open", "decision": "new", "clv_pct": 13.846154})
            migration = Path(__file__).parents[1] / "migrations" / "1823_historical_clv.sql"
            cur.execute("CREATE ROLE anon; CREATE ROLE authenticated; CREATE ROLE service_role")
            cur.execute(migration.read_text())
            cur.execute(migration.read_text())
            self.assertEqual(cur.execute("SELECT relrowsecurity FROM pg_class "
                                         "WHERE relname='fb_historical_clv_1823'").fetchone(), (True,))
            for role in ("anon", "authenticated", "service_role"):
                self.assertEqual(cur.execute("SELECT has_table_privilege(%s,'public.fb_historical_clv_1823','SELECT')",
                                             (role,)).fetchone(), (False,))
            c.commit()
            cur.execute('SAVEPOINT one')
            cur.execute('UPDATE public.football_bets SET "Initial_Line"=10 WHERE "ID"=%s', (rows[0]["bid"],))
            self.assertEqual(cur.execute(script(rows, with_table=True, approved=planned[0])).fetchone()[-1], 0)
            cur.execute('ROLLBACK TO one')
            eng = create_engine(URL.create("postgresql+psycopg", database=info["dbname"],
                                           query={"host": info["host"], "port": info["port"]}))
            self.assertEqual(cur.execute(script(rows, with_table=True, approved=planned[0])).fetchone()[-1], 708)
            self.assertEqual(cur.execute(script(rows, with_table=True, approved=planned[0])).fetchone()[-1], 0)
            self.assertEqual(cur.execute("SELECT count(*) FROM public.fb_historical_clv_1823").fetchone(), (708,))
            self.assertEqual(cur.execute("SELECT md5(string_agg(row_to_json(b)::text,'' ORDER BY \"ID\")) "
                                         "FROM public.football_bets b").fetchone()[0], before)
            self.assertEqual(cur.execute("SELECT evidence->>'closing_url' FROM public.fb_historical_clv_1823 "
                                         "WHERE bet_id=%s", (rows[0]["bid"],)).fetchone()[0],
                             rows[0]["provenance"]["closing_url"])
            c.commit()
            with eng.connect() as conn:
                evidence = sql_rows(conn)
            df = pd.DataFrame(evidence)
            baseline = pd.DataFrame({"group": "historical_unknown", "clv": float("nan"),
                                     "closing": float("nan"), "close_time": None, "reason": None}, index=df.index)
            out, official, visible = overlay(df, baseline, evidence)
            self.assertEqual(len(visible), 708)
            done = official.loc[out.Status.eq("Settled"), "clv"]
            self.assertEqual(len(done), 681)
            self.assertAlmostEqual(done.mean()*100, .070426, places=6)
            self.assertEqual((done.gt(0).sum(), done.eq(0).sum(), done.lt(0).sum()), (173, 331, 177))
            cur.execute('INSERT INTO public.football_bets_deleted VALUES (%s)', (rows[0]["bid"],))
            c.commit()
            with eng.connect() as conn:
                self.assertIsNone(next(r for r in sql_rows(conn) if r["bet_id"] == rows[0]["bid"])["ID"])
            eng.dispose()
