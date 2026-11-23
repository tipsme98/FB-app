"""CLV provenance, archive integration and actual Streamlit renderer (synthetic, offline)."""

import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from fb_app_clv_cohorts import ENTRY, HKT, QUOTE, classify, entry_note, fixture_id, local, quote_tag, report_period

APP = Path(__file__).resolve().parents[1] / "FB-app.py"
KICK = datetime(2026, 11, 23, 0, 10, tzinfo=HKT)
ENTRY_AT, OBSERVED, QUOTED = (KICK - timedelta(minutes=x) for x in (32, 33, 35))


def base(**overrides):
    return dict({
        "ID": "B999000", "Tipsme_Match_ID": 999001, "Bet_Type": "入球大小", "Selection": "Under",
        "Initial_Line": 2.75, "Initial_Odds": 1.95, "Match_Date": KICK.date().isoformat(),
        "Date": ENTRY_AT.strftime("%Y-%m-%d %H:%M"), "Odds_History": "[]", "Status": "Settled",
        "System_Stake": 0.0, "User_Stake": 0.0, "Match": "合成場", "Note": "",
        "System_Profit": 0.0, "User_Profit": 0.0, "User_Payout": 0.0,
        "Unit_Profit": 0.0, "Result_Label": "走盤",
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
    def test_hong_kong_cutoff_and_unknown_dates(self):
        cases = {"2026-11-21 23:59:59": 0, "2026-11-22T00:00:00+08:00": 1,
                 "2026-11-21T16:00:00Z": 1, "2026-11-22T00:00:00+09:00": 0,
                 "2026-11-22": None, "broken": None, None: None, pd.NaT: None}
        for value, expected in cases.items():
            self.assertEqual(report_period(value), expected)

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

    def test_diagnostic_reasons_do_not_relax_eligibility(self):
        for q, reason in (
            (quote(ENTRY_AT), "not_after_entry"),
            (quote(KICK - timedelta(minutes=16)), "too_early"),
            (quote(KICK - timedelta(minutes=2), line=2.5), "line_changed"),
            (quote(KICK - timedelta(minutes=2), close=False), "no_archive"),
            (quote(KICK - timedelta(minutes=2), mid=2), "no_archive"),
        ):
            with self.subTest(reason=reason):
                result = classify(marked(q))
                self.assertEqual((result.group, result.clv, result.reason), ("pending", None, reason))
        boundary = classify(marked(quote(KICK - timedelta(minutes=15))))
        self.assertEqual((boundary.group, boundary.clv), ("snapshot", 0.0))

    def test_same_final_source_quote_and_counterexamples(self):
        result = classify(marked(quote(QUOTED)))
        self.assertEqual((result.group, result.clv), ("snapshot", 0.0))
        self.assertEqual(classify(marked(quote(QUOTED, close=False))).group, "pending")
        self.assertEqual(classify(marked({**quote(QUOTED), "upper": 1.91})).group, "pending")
        self.assertEqual(classify(marked(quote(QUOTED - timedelta(seconds=1)))).group, "pending")
        self.assertEqual(classify(marked(quote(QUOTED), quote(QUOTED, line=2.5))).group, "close_conflict")

    def test_manual_entry_only_unchanged_fresh_actual_latest_provider_snapshot(self):
        scope = dict(clv_fixture_id=fixture_id, clv_entry_note=entry_note)
        exec(compile(ast.Module(body=nodes({"_clv_manual_entry_note"}), type_ignores=[]), str(APP), "exec"), scope)
        fn = scope["_clv_manual_entry_note"]
        fill = {**locked()["fill"], "clv_manual_prices": locked()["odds"], "clv_manual_kickoff": KICK.timestamp()}
        self.assertTrue(fn(base(), CANDIDATE, fill, locked()["odds"], ENTRY_AT))
        reconstructed = {k: v for k, v in locked()["odds"][0].items() if k != QUOTE}
        self.assertTrue(fn(base(), CANDIDATE, fill, [reconstructed], ENTRY_AT))
        self.assertIsNone(fn(base(), CANDIDATE, fill, locked()["odds"], ENTRY_AT+timedelta(minutes=6)))
        self.assertIsNone(fn(base(), CANDIDATE, fill, [{**reconstructed, "upper": 1.90}], ENTRY_AT))
        self.assertIsNone(fn(base(), CANDIDATE, fill, [*locked()["odds"], quote(ENTRY_AT)], ENTRY_AT))
        self.assertIsNone(fn(base(Tipsme_Match_ID=2), CANDIDATE, fill, locked()["odds"], ENTRY_AT))
        self.assertIsNone(fn(base(), CANDIDATE, {**fill, "clv_manual_kickoff": KICK.timestamp()+60},
                             locked()["odds"], ENTRY_AT))

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

    def test_same_price_after_kickoff_does_not_erase_certified_final_quote(self):
        raw = self.raw((QUOTED.isoformat(), 2.75, 1.95),
                       ((KICK+timedelta(minutes=2)).isoformat(), 2.75, 1.95))
        history = self.archive(raw)
        self.assertEqual([q[QUOTE][1] for q in history if QUOTE in q], [QUOTED.isoformat()])
        self.assertEqual(classify(marked(*history)).clv, 0.0)
        ambiguous = self.raw((QUOTED.isoformat(), 2.75, 1.95), (QUOTED.isoformat(), 2.5, 1.90))
        self.assertFalse(any(QUOTE in q for q in self.archive(ambiguous)))

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
    shared = {"CLV_GROUPS", "_V16_CLV_HOLDER", "_v16_db_content_sig", "_v16_clv_rows"}

    def boot(self, rows, *, full=False):
        names = self.shared | {"render_clv_section"}
        if full:
            names |= {"render_clv_tab", "render_clv_summary", "render_clv_period_stats", "render_clv_recent_diagnostics",
                      "render_handicap_audit", "compute_legacy_bet_clv",
                      "v9_clean_bt", "_oh_sort_ts", "parse_dt_for_comparison", "v10_parse_time",
                      "get_hkt_now", "HKT", "render_monte_carlo_section", "build_mc_records",
                      "run_bankroll_simulation"}
        wrapper = ("import pandas as pd\nimport numpy as np\nimport streamlit as st\nimport json, re\n"
                   "from datetime import datetime, timedelta, timezone\n"
                   "from fb_app_clv_cohorts import classify as classify_clv, ENTRY as CLV_ENTRY, local as clv_local, report_period as clv_report_period\n"
                   "from fb_app_handicap_audit import handicap_audit\n"
                   + "\n".join(ast.unparse(n) for n in nodes(names))
                   + f"\nframe = pd.DataFrame({rows!r}, index=list(range(719, 719+{len(rows)})))\n"
                   + ("def render_clv_closing_updater(*_): st.button('收盤更新測試替身')\n"
                      "render_clv_tab(frame, None, None, 10000)\n" if full else "render_clv_section(frame)\n")
                   + "st.session_state['groups'] = _v16_clv_rows(frame).group.tolist()\n"
                   + "assert _v16_clv_rows(frame) is _v16_clv_rows(frame)\n"
                   + "if len(frame):\n"
                   + "    changed = frame.copy(); changed['Note'] = '用家跟投'\n"
                   + "    assert _v16_clv_rows(changed).group.eq('follow').all()\n")
        at = AppTest.from_string(wrapper).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        return at

    def test_reports_empty_as_missing_and_legitimate_zero_as_zero(self):
        legacy = base(ID="legacy", Date="2026-11-24 11:00")
        at = self.boot([legacy])
        self.assertEqual([m.value for m in at.metric], ["0 注", "無估計", "無樣本"])
        at = self.boot([legacy, marked(quote(KICK-timedelta(minutes=2)))])
        self.assertEqual([m.value for m in at.metric], ["1 注", "+0.00%", "1 注"])
        self.assertEqual(at.session_state["groups"], ["historical_late", "snapshot"])

    def test_single_view_keeps_archive_outside_official_counts_and_preserves_order(self):
        def history(close):
            # Reverse storage order to check date ordering; wrong lines must not be used.
            return json.dumps([quote(KICK-timedelta(minutes=1), price=close),
                               quote(KICK, price=1.10, line=3),
                               quote(QUOTED)])

        rows = [
            base(ID="old", Date="2026-10-20 23:00", Match_Date="2026-10-20", Match="較早紀錄",
                 Odds_History=history(1.95), Unit_Profit=-1),
            base(ID="new", Match="較新紀錄", Odds_History=history(1.80), Unit_Profit=.95,
                 System_Stake=10, System_Profit=9.5, Result_Label="全贏"),
            base(ID="pending", Status="Pending", Odds_History=history(1.80)),
            base(ID="single", Odds_History=json.dumps([quote(QUOTED)])),
            base(ID="live", Bet_Type="入球大小 (即場)", Odds_History=history(1.80)),
            base(ID="bad-date", Date="2026-11-22", Odds_History=history(1.80)),
            marked(quote(QUOTED), Match="可證明同價"),
        ]
        at = self.boot(rows, full=True)
        headings = [m.value for m in at.markdown]
        order = ["#### CLV (收盤線價值) 追蹤", "##### 分盤口類型 CLV 統計",
                 "##### CLV 與賽果的關係", "##### 平均 CLV 按月趨勢",
                 "##### 最近 15 注 CLV 明細", "#### 🎲 蒙地卡羅本金模擬", "#### 賽前快照證據分組（V18.1）"]
        self.assertEqual(sorted(headings.index(heading) for heading in order),
                         [headings.index(heading) for heading in order])
        self.assertEqual(headings.count("##### 最近 15 注 CLV 明細"), 1)
        self.assertEqual(len(at.tabs), 0)
        self.assertEqual([m.value for m in at.metric if m.label == "可計算 CLV 的注單"], ["1 注"])
        self.assertEqual([m.value for m in at.metric if m.label == "舊版可比對紀錄"], ["2 注"])
        self.assertEqual([m.value for m in at.metric if m.label == "平均 CLV"], ["+0.00%"])
        price_details = [d.value for d in at.dataframe if "價格差%" in d.value.columns]
        self.assertEqual(price_details[0]["Match"].tolist(), ["較新紀錄", "較早紀錄"])
        self.assertEqual(price_details[0]["價格差%"].tolist(), [8.33, 0.0])
        evidence = next(d.value for d in at.dataframe if "全部注數" in d.value.columns)
        self.assertEqual(evidence["全部注數"].sum(), 7)
        self.assertEqual(evidence["已結算"].sum(), 6)
        at.selectbox(key="mc_account").select("User").run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertTrue(any("User 帳戶" in m.value for m in at.info))

    def test_empty_ledger_shows_only_one_set_of_headings(self):
        at = self.boot([], full=True)
        self.assertEqual(len(at.tabs), 0)
        self.assertEqual(sum(m.value == "##### 平均 CLV 按月趨勢" for m in at.markdown), 1)
        self.assertEqual([m.value for m in at.metric if m.label == "可計算 CLV 的注單"], ["0 注"])
        self.assertTrue(any("無有效登記時間的紀錄" in m.value for m in at.info))

    def test_recent_diagnostics_read_only_status_and_evidence_counts(self):
        final = quote(KICK-timedelta(minutes=2))
        rows = [
            base(Note="", Match="未有標記"),
            marked(Match="缺收盤"),
            marked(quote(KICK-timedelta(minutes=2), line=2.5), Match="轉線"),
            marked(final, Status="Open", Match="未結算"),
            marked(final, Match="已結算"),
            marked(final, Status="Unknown", Match="其他"),
            base(Date="2026-11-21 14:00"),
            base(Date="2026-11-22"),
        ]
        at = self.boot(rows, full=True)
        self.assertIn("近期帳本 6 筆｜已結算 4 筆｜待結算 1 筆｜其他狀態 1 筆",
                      [m.value for m in at.markdown])
        t = next(el.value for el in at.table if "核對結果" in el.value.columns)
        self.assertEqual(t[["全部", "已結算", "待結算", "其他"]].sum().tolist(), [6, 4, 1, 1])
        good = t[t["核對結果"].str.startswith("入注及收盤證據齊全")].iloc[0]
        self.assertEqual(good[["全部", "已結算", "待結算", "其他"]].tolist(), [3, 1, 1, 1])
        self.assertEqual([m.value for m in at.metric if m.label == "可計算 CLV 的注單"], ["1 注"])

    def test_preview_excel_and_csv_keep_missing_distinct_from_zero(self):
        preview, = nodes({"preview_db_dialog"})
        # Run the actual bets tab, avoiding unrelated management and any data writes.
        preview.body = preview.body[:2]
        preview.decorator_list = []
        names = self.shared | {"_v16_search_mask", "_V16_LIGHT_SEARCH_COLS", "_V16_JSON_HEAVY_COLS"}
        rows = [base(ID="old", Odds_History=json.dumps([quote(QUOTED), quote(ENTRY_AT)])),
                marked(quote(KICK-timedelta(minutes=2)))]
        setup = ("import io\nimport pandas as pd\nimport streamlit as st\n"
                   "from unittest.mock import patch\n"
                   "from fb_app_clv_cohorts import classify as classify_clv, ENTRY as CLV_ENTRY\n"
                   + "\n".join(ast.unparse(n) for n in [*nodes(names), preview])
                   + f"\nframe = pd.DataFrame({rows!r})\n")
        for fmt, reader in (("xlsx", "read_excel"), ("csv", "read_csv")):
            force_csv = ", patch.object(pd.DataFrame, 'to_excel', side_effect=RuntimeError)" if fmt == "csv" else ""
            run = (f"with patch.object(st, 'download_button') as download{force_csv}:\n"
                   "    preview_db_dialog(frame, None, None, None, None, None)\n"
                   "    args = download.call_args.args\n"
                   f"    assert args[2].endswith('.{fmt}')\n"
                   f"    st.session_state['export'] = pd.{reader}(io.BytesIO(args[1]))\n")
            at = AppTest.from_string(setup + run).run()
            self.assertFalse(at.exception)
            for table in (at.dataframe[0].value, at.session_state["export"]):
                self.assertTrue(pd.isna(table.loc[0, "CLV%"]))
                self.assertEqual(table.loc[1, "CLV%"], 0.0)
                self.assertEqual(table.loc[1, "CLV_組別"], "可核對賽前快照")


if __name__ == "__main__":
    unittest.main()
