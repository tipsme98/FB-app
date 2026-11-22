"""CLV provenance, archive integration and actual Streamlit renderer (synthetic, offline)."""

import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from fb_app_clv_cohorts import ENTRY, HKT, QUOTE, classify, entry_note, fixture_id, local, quote_tag

APP = Path(__file__).resolve().parents[1] / "FB-app.py"
KICK = datetime(2026, 11, 23, 0, 10, tzinfo=HKT)
ENTRY_AT, OBSERVED, QUOTED = (KICK - timedelta(minutes=x) for x in (32, 33, 35))


def base(**overrides):
    return dict({
        "ID": "B999000", "Tipsme_Match_ID": 999001, "Bet_Type": "入球大小", "Selection": "Under",
        "Initial_Line": 2.75, "Initial_Odds": 1.95, "Match_Date": KICK.date().isoformat(),
        "Date": ENTRY_AT.strftime("%Y-%m-%d %H:%M"), "Odds_History": "[]", "Status": "Settled",
        "System_Stake": 0.0, "User_Stake": 0.0, "Match": "合成場", "Note": "",
        "Home_Team": "主隊", "Away_Team": "客隊",
    }, **overrides)


def quote(time, *, close=True, price=1.95, line=2.75, mid=999001):
    return {"type": "入球大小", "line": line, "upper": 1.85, "lower": price,
            "record_time": time.strftime("%m-%d %H:%M"),
            QUOTE: quote_tag(mid, time, KICK if close else None)}


def locked():
    return dict(token=("sid", 0, "999001"), odds=[quote(QUOTED, close=False)],
                fill=dict(info=dict(match_id="999001", kickoff=KICK, finished=False, clv_id_confirmed=True),
                          odds_ok=True, clv_observed=OBSERVED.isoformat()))


CANDIDATE = {"history_idx": 0, "is_latest": True}


def marked(*quotes, **overrides):
    row = base()
    row["Note"] = entry_note(row, CANDIDATE, locked(), KICK.timestamp(), ENTRY_AT)
    assert row["Note"] and row["Note"].startswith(ENTRY)
    row.update(Odds_History=json.dumps(quotes), **overrides)
    return row


def nodes(names):
    return [n for n in ast.parse(APP.read_text()).body
            if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names)]


class Evidence(unittest.TestCase):
    def test_legacy_never_inherits_price_proof(self):
        self.assertEqual(classify(base(Date="2026-11-24 02:00")).group, "historical_late")
        self.assertEqual(classify(base(Date="2026-11-23 00:08")).group, "historical_unknown")
        self.assertEqual(classify(base()).group, "historical_unknown")
        self.assertEqual(classify(base(Note="用家跟投")).group, "follow")
        self.assertEqual(classify(base(Bet_Type="入球大小 (即場)")).group, "out_of_scope")

    def test_independent_zero_and_direction(self):
        r = classify(marked(quote(KICK-timedelta(minutes=2))))
        self.assertEqual((r.group, r.clv), ("snapshot", 0.0))
        r = classify(marked(quote(KICK-timedelta(minutes=2), price=1.80)))
        self.assertAlmostEqual(r.clv, 1.95/1.80-1)

    def test_no_independent_comparator(self):
        for q in (quote(ENTRY_AT), quote(KICK), quote(KICK+timedelta(seconds=1)),
                  quote(KICK-timedelta(minutes=16)), quote(KICK-timedelta(minutes=2), mid=2),
                  quote(KICK-timedelta(minutes=2), line=2.5), quote(KICK-timedelta(minutes=2), close=False)):
            with self.subTest(q=q):
                self.assertEqual(classify(marked(q)).group, "pending")

    def test_latest_line_and_ambiguities(self):
        first = quote(KICK-timedelta(minutes=5))
        last = quote(KICK-timedelta(minutes=2), line=2.5)
        self.assertEqual(classify(marked(last, first)).group, "pending")
        last["line"] = 2.75
        other = {**last, "lower": 1.88}
        self.assertEqual(classify(marked(last, first, other)).group, "close_conflict")
        last[QUOTE][2] = (KICK+timedelta(hours=2)).isoformat()
        self.assertEqual(classify(marked(last, first)).group, "kickoff_conflict")

    def test_marker_binding_and_bad_inputs(self):
        for patch in ({"Initial_Odds": 1.96}, {"Selection": "Over"}, {"ID":"changed"},
                      {"Date": "2026-11-23 02:00"}, {"Note": ENTRY + "[[]]"}):
            self.assertEqual(classify(marked(**patch)).group, "entry_invalid")
        self.assertIsNone(entry_note(base(), {"history_idx": None}, locked(), KICK.timestamp(), ENTRY_AT))
        self.assertIsNone(entry_note(base(), CANDIDATE, locked(), KICK.timestamp()+600, ENTRY_AT))
        self.assertIsNone(entry_note(base(Date=KICK.isoformat()), CANDIDATE, locked(), KICK.timestamp(), KICK))
        p = locked()
        p["odds"][0][QUOTE][1] = (OBSERVED+timedelta(seconds=1)).isoformat()
        self.assertIsNone(entry_note(base(), CANDIDATE, p, KICK.timestamp(), ENTRY_AT))
        p["odds"][0][QUOTE][1] = QUOTED.strftime("%Y-%m-%dT%H:%M:%S")
        self.assertIsNone(entry_note(base(), CANDIDATE, p, KICK.timestamp(), ENTRY_AT))


class Archive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = {"_tipsme_market_records", "_tipsme_oh_entry", "_tipsme_iso_time", "_tipsme_cap_value",
                 "_tipsme_pool_recs", "_tipsme_add_market", "tipsme_build_odds_history", "_clv_build_archive",
                 "merge_odds_history_snapshots", "clv_apply_closing_odds"}
        cls.scope = dict(pd=pd, re=re, json=json, datetime=datetime, timedelta=timedelta,
                         CLV_QUOTE=QUOTE, CLV_ENTRY=ENTRY, clv_local=local, clv_quote_tag=quote_tag,
                         clv_fixture_id=fixture_id, _oh_sort_ts=lambda h, i: h.get("record_time", ""),
                         is_missing_match_date=lambda d: not d)
        exec(compile(ast.Module(body=nodes(names), type_ignores=[]), str(APP), "exec"), cls.scope)

    def raw(self, *quotes):
        return {"hiloOdds": [
            {"create_time": t, "number_of_goals": line, "big_odds": 1.85, "small_odds": price}
            for t, line, price in quotes]}

    def archive(self, raw, **info):
        return self.scope["_clv_build_archive"](raw, 999001,
                                               dict({"match_id": 999001, "kickoff": KICK, "finished": True,
                                                     "clv_id_confirmed": True}, **info))[0]

    def test_time_zone_preserved_and_only_final_market_line_tagged(self):
        t1, t2, t3 = ((KICK+timedelta(minutes=i)).astimezone(timezone.utc).isoformat() for i in (-5, -2, 3))
        raw = self.raw((t1, 2.75, 1.95), (t2, 2.5, 1.9), (t3, 2.75, 1.88))
        quotes = self.archive(raw)
        tags = [q for q in quotes if QUOTE in q]
        self.assertEqual([q["line"] for q in tags], [2.5])
        self.assertTrue(tags[0][QUOTE][1].endswith("+08:00"))
        self.assertEqual(classify(marked(*quotes)).group, "pending")
        self.assertFalse(any(QUOTE in q for q in self.archive(raw, match_id=999002)))
        self.assertFalse(any(QUOTE in q for q in self.archive(raw, finished=False)))
        self.assertFalse(any(QUOTE in q for q in self.archive(raw, clv_id_confirmed=False)))

    def test_missing_zone_at_final_update_does_not_backfill_earlier(self):
        t1, t2 = ((KICK+timedelta(minutes=i)).astimezone(timezone.utc).isoformat() for i in (-5, -2))
        raw = self.raw((t1, 2.75, 1.95), (t2[:19], 2.75, 1.88))
        self.assertFalse(any(QUOTE in q for q in self.archive(raw)))

    def test_merge_does_not_change_price_and_only_marks_identified_new_entries(self):
        new = quote(KICK-timedelta(minutes=2))
        previous = {k: v for k, v in new.items() if k != QUOTE}
        frame = pd.DataFrame([marked(previous), base(ID="legacy", Odds_History=json.dumps([previous])),
                              marked(previous, Tipsme_Match_ID=999002)])
        fn = self.scope["clv_apply_closing_odds"]
        self.assertEqual(fn(frame, [new], "主隊", "客隊", KICK)[1], 1)
        self.assertEqual(fn(frame, [new], "主隊", "客隊", KICK)[1], 0)
        self.assertEqual(classify(frame.iloc[0]).clv, 0)
        self.assertNotIn(QUOTE, frame.iloc[1].Odds_History)
        self.assertNotIn(QUOTE, frame.iloc[2].Odds_History)
        self.assertEqual(list(frame.Initial_Odds), [1.95]*3)


class StreamlitPanel(unittest.TestCase):
    def boot(self, rows):
        names = {"CLV_GROUPS", "_V16_CLV_HOLDER", "_v16_db_content_sig", "_v16_clv_rows", "render_clv_section"}
        wrapper = ("import pandas as pd\nimport streamlit as st\n"
                   "from fb_app_clv_cohorts import classify as classify_clv, ENTRY as CLV_ENTRY\n"
                   + "\n".join(ast.unparse(n) for n in nodes(names))
                   + f"\nframe = pd.DataFrame({rows!r}, index=list(range(719, 719+{len(rows)})))\n"
                   + "render_clv_section(frame)\n"
                   + "st.session_state['groups'] = _v16_clv_rows(frame).group.tolist()\n"
                   + "assert _v16_clv_rows(frame) is _v16_clv_rows(frame)\n"
                   + "if len(frame):\n"
                   + "    changed = frame.copy(); changed['Note'] = '用家跟投'\n"
                   + "    assert _v16_clv_rows(changed).group.eq('follow').all()\n")
        at = AppTest.from_string(wrapper).run()
        self.assertFalse(at.exception)
        return at

    def test_reports_empty_as_missing_and_legitimate_zero_as_zero(self):
        legacy = base(ID="legacy", Date="2026-11-24 11:00")
        at = self.boot([legacy])
        self.assertEqual([m.value for m in at.metric], ["0 注", "無估計", "無樣本"])
        at = self.boot([legacy, marked(quote(KICK-timedelta(minutes=2)))])
        self.assertEqual([m.value for m in at.metric], ["1 注", "+0.00%", "1 注"])
        self.assertEqual(at.session_state["groups"], ["historical_late", "snapshot"])


if __name__ == "__main__":
    unittest.main()
