"""Offline tests for the actual follow-tab renderer, using synthetic slips only."""
import ast
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest
from fb_app_profitable_rules import freeze

APP = Path(__file__).resolve().parents[1] / "FB-app.py"
FOLLOW_FUNCTIONS = {
    "v10_md_series", "v10_match_groups", "v10_match_mask",
    "_v162_follow_record_time", "_v162_follow_match_sections", "v162_render_follow_tab",
    "_follow_auto_width", "_follow_row_style",
}
FOLLOW_CONSTANTS = {"FOLLOW_COLUMN_ORDER", "FOLLOW_CANDIDATE_STYLE"}

PRELUDE = """
from datetime import datetime, timedelta, timezone
import re
import pandas as pd
import streamlit as st

HKT = timezone(timedelta(hours=8))
V162_USER_BET_NOTE = '用家跟投'
get_hkt_now = lambda: datetime(2026, 11, 21, 22, 31, 3, tzinfo=HKT)
_v162_runner_mode_on = lambda: True
_v162_is_runner = lambda: False
_v162_hb_fresh = lambda: True
_v161_refresh_db_from_sql = lambda: None
_follow_analysis = lambda df: None
_follow_rule_states = lambda *a, **k: {}
def save_db(df, filename, table):
    assert filename == 'football_betting_db.csv' and table == 'football_bets'
    st.session_state['test_writes'] = st.session_state.get('test_writes', 0) + 1
"""


def slip(name, bet_id, recorded, *, day="2026-11-21", system=0, status="Open"):
    return {
        "Tournament_Name": "離線測試聯賽", "Match": name, "Match_Date": day,
        "ID": bet_id, "Date": recorded, "System_Stake": system, "User_Stake": 0,
        "Status": status, "Note": "", "Bet_Type": "入球大小", "Selection": "Over",
        "Initial_Line": 2.5, "Initial_Odds": 1.91,
        "Model_Prob": 0.55, "Market_Prob": 0.52, "Edge": 0.03, "Gate": "PASS",
    }


def unit(fn, expr):
    nodes = [n for n in ast.parse(APP.read_text(encoding="utf-8")).body
             if (isinstance(n, ast.FunctionDef) and n.name == fn)
             or (isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in FOLLOW_CONSTANTS for t in n.targets))]
    src = PRELUDE + "\n" + "\n\n".join(ast.unparse(n) for n in nodes)
    src += f"\nst.session_state['out'] = {expr}\n"
    return AppTest.from_string(src).run()


def boot(*rows):
    nodes = [n for n in ast.parse(APP.read_text(encoding="utf-8")).body
             if (isinstance(n, ast.FunctionDef) and n.name in FOLLOW_FUNCTIONS)
             or (isinstance(n, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in FOLLOW_CONSTANTS for t in n.targets))]
    wrapper = PRELUDE + "\n" + "\n\n".join(ast.unparse(n) for n in nodes)
    wrapper += (
        "\nif 'df_db' not in st.session_state:\n"
        f"    st.session_state['df_db'] = pd.DataFrame({list(rows)!r})\n"
        "with st.tabs(['用家跟投'])[0]:\n    v162_render_follow_tab()\n"
    )
    return AppTest.from_string(wrapper).run()


class FollowTableDefaults(unittest.TestCase):
    def test_default_columns_follow_requested_order(self):
        at = boot(slip("預設欄", "B202611212100000", "2026-11-21 21:00"))
        self.assertFalse(at.exception)
        cols = list(at.dataframe[0].value.columns)
        self.assertEqual(cols, [
            "Bet_Type", "Selection", "Initial_Line", "Initial_Odds", "System_Stake", "歷史樣本",
            "歷史 ROI", "命中率", "單位盈虧", "狀態", "反向單位盈虧", "反向ROI",
            "反向命中率", "看盤規則", "Model_Prob", "Market_Prob", "Edge", "Gate",
            "User_Stake"])
        self.assertNotIn("ID", cols)
        self.assertEqual(cols.index("Initial_Line"), cols.index("Selection") + 1)

    def test_widths_grow_for_cjk_and_clamp(self):
        at = unit("_follow_auto_width", "[_follow_auto_width('G', ['1']), "
                  "_follow_auto_width('狀態', ['模擬候選']), _follow_auto_width('x', ['y' * 400])]")
        self.assertFalse(at.exception)
        short, cjk, long = at.session_state["out"]
        self.assertEqual(short, 84)
        self.assertGreater(cjk, short)
        self.assertEqual(long, 360)

    def test_candidate_status_highlights_every_cell(self):
        at = unit("_follow_row_style", "[_follow_row_style({'狀態': '模擬候選', 'Bet_Type': '讓球', "
                  "'Gate': 'x'}), _follow_row_style({'狀態': '未入選', 'Bet_Type': '讓球'})]")
        self.assertFalse(at.exception)
        candidate, other = at.session_state["out"]
        self.assertEqual(len(candidate), 3)
        self.assertTrue(all(candidate))
        self.assertEqual(set(candidate), {"background-color:#d9eee9;color:#163b38"})
        self.assertFalse(any(other))


class FollowNewestFirst(unittest.TestCase):
    def test_frozen_candidate_is_labelled_and_old_row_remains_blank(self):
        new = slip("新場", "B1", "2026-11-21 21:00")
        old = slip("新場", "B2", "2026-11-21 20:00")
        removed = slip("新場", "B3", "2026-11-21 19:00")
        new["Odds_History"] = "[{}]"
        removed["Odds_History"] = "[{}]"
        freeze(new, None, {"rule": "盤水不變", "n": 60, "roi": .08, "hit": .60})
        freeze(removed, {"rule": "收盤高水", "n": 70, "roi": -.02, "hit": .51,
                         "candidate_status": "移出／待重入"}, None)
        at = boot(new, old, removed)
        self.assertFalse(at.exception)
        view = at.dataframe[0].value
        candidate = view[view["狀態"] == "模擬候選"]
        self.assertEqual(len(candidate), 1)
        self.assertEqual(candidate.iloc[0]["看盤規則"], "盤水不變")
        blank = view[view["看盤規則"] == "—"]
        self.assertEqual(len(blank), 1)
        self.assertEqual(blank.iloc[0]["狀態"], "")
        self.assertEqual(list(view["狀態"]).count("移出／待重入"), 1)

    def assert_order(self, at, *matches):
        self.assertFalse(at.exception)
        labels = [e.label for e in at.get("expander")]
        self.assertEqual(len(labels), len(matches))
        for label, match in zip(labels, matches):
            self.assertIn(match, label)
        return labels

    def test_recent_entry_precedes_old_real_bet_and_ignores_kickoff_and_settled_rows(self):
        at = boot(
            slip("舊實注場", "B202611202019000", "2026-11-20 20:19", day="2026-11-22", system=100),
            slip("舊實注場", "B202611212230000", "2026-11-21 22:30", day="2026-11-22", status="Settled"),
            slip("新模擬場", "B202611212100000", "2026-11-21 21:00", day="2026-11-20"),
            slip("已結算場", "B202611212231000", "2026-11-21 22:31", status="Settled"),
        )
        labels = self.assert_order(at, "新模擬場", "舊實注場")
        self.assertIn("模擬注 ($0)", labels[0])
        self.assertIn("系統實注", labels[1])
        self.assertNotIn("test_writes", at.session_state)

    def test_group_recency_uses_latest_distinct_open_record_and_id_seconds(self):
        at = boot(
            slip("A 較早", "B202611212120100", "2026-11-21 21:20"),
            slip("Z 較新", "B202611211500100", "2026-11-21 15:00"),
            slip("Z 較新", "IP202611212120510", "2026-11-21 21:20"),
            slip("A 較早", "B202611212120100", "2026-11-21 23:59"),
        )
        labels = self.assert_order(at, "Z 較新", "A 較早")
        self.assertIn("2 模擬", labels[0])
        self.assertIn("1 模擬", labels[1])
        self.assertEqual(len(at.checkbox), 3)
        self.assertTrue(all(button.label == "✅ 寫入用家跟投注單" for button in at.button))

    def test_invalid_times_last_and_same_fixture_on_two_dates_stays_separate(self):
        at = boot(
            slip("同一組對賽", "B202611212100000", "2026-11-21 21:00", day="2026-11-20"),
            slip("同一組對賽", "B202611212011000", "2026-11-21 20:11", day="2026-11-21"),
            slip("無時間的實注", "old", "bad clock", system=100),
        )
        labels = self.assert_order(at, "同一組對賽", "同一組對賽", "無時間的實注")
        self.assertIn("2026-11-20", labels[0])
        self.assertIn("2026-11-21", labels[1])

    def test_automatic_id_microseconds_and_conflicting_id_minute(self):
        earlier_id = "B20261121212035" + "190000" + "0" * 14
        later_id = "B20261121212035" + "890000" + "0" * 14
        at = boot(
            slip("A 較早微秒", earlier_id, "2026-11-21 21:20"),
            slip("Z 較後微秒", later_id, "2026-11-21 21:20"),
            slip("日期與 ID 不符", "B202611212121590", "2026-11-21 21:20"),
        )
        self.assert_order(at, "Z 較後微秒", "A 較早微秒", "日期與 ID 不符")

    def test_follow_button_still_updates_only_its_match_and_promotes_new_input(self):
        at = boot(
            slip("先前較新", "B202611212215000", "2026-11-21 22:15"),
            slip("現在跟投", "B202611212200000", "2026-11-21 22:00"),
        )
        self.assert_order(at, "先前較新", "現在跟投")
        at.button(key="v162_go_2026-11-21_現在跟投").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("請先勾選注項" in notice.value for notice in at.warning))
        at.checkbox(key="v162_ck_0_B202611212200000").set_value(True)
        at.number_input(key="v162_stk_0_B202611212200000").set_value(50.0)
        at.button(key="v162_go_2026-11-21_現在跟投").click().run()
        labels = self.assert_order(at, "現在跟投", "先前較新")
        self.assertIn("用家已跟投 1 注", labels[0])
        self.assertEqual(at.session_state["test_writes"], 1)
        fresh = at.session_state["df_db"].iloc[-1]
        self.assertEqual((fresh["Match"], fresh["User_Stake"]), ("現在跟投", 50.0))


if __name__ == "__main__":
    unittest.main()
