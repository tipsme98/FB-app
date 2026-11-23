"""Offline review for the scoped V10 repair and the read-only market benchmark."""

import ast
from datetime import timedelta
import unittest

from streamlit.testing.v1 import AppTest

from fb_app_clv_cohorts import entry_note, record_handicap_forecast
from fb_app_handicap_audit import handicap_audit
from test_clv_cohorts import APP, CANDIDATE, ENTRY_AT, KICK, base, locked


KEYS = ("v11_new_league_paper", "v11_league_breaker", "v11_auto_sys")
V10_NAMES = {"V10_MODE_OPTIONS", "V10_SETTING_DEFAULTS", "V11_LS_PREFIX", "_V10_META_KEYS",
             "_V10_RESTORE_KEYS", "_V10_LIMITS", "_V10SettingsConflict", "_v10_cast",
             "_v10_key_values", "_v10_patch_rows", "_v10_csv_rows", "_v10_csv_content",
             "_v10_apply_aux", "_v10_apply_settings", "v10_load_settings_into_state",
             "_v10_settings_write", "v10_save_settings", "v10_restore_risk_controls",
             "_v10_settings_allow_automation"}


class V10Settings(unittest.TestCase):
    def boot(self, rows=(), *, fail=False):
        nodes = [n for n in ast.parse(APP.read_text()).body
                 if getattr(n, "name", None) in V10_NAMES
                 or (isinstance(n, ast.Assign) and (getattr(n.targets[0], "id", "") in V10_NAMES
                     or getattr(n.targets[0], "id", "").startswith(("V11_MODE_", "V10_MODE_"))))]
        at = AppTest.from_string(
            "import streamlit as st\n"
            + "\n".join(map(ast.unparse, nodes)) + "\n"
            + f"st.session_state.setdefault('stores', {{'Supabase': {list(rows)!r}, 'GitHub': []}})\n"
            + f"st.session_state.setdefault('fail', {fail!r})\n"
            + """
st.session_state.setdefault('source', 'Supabase')
st.session_state.setdefault('writes', [])
st.session_state.setdefault('reads', [])
def _v10_settings_source():
    return st.session_state.source
def _v10_settings_store(source, changes=None, expected=None):
    st.session_state.reads.append(source)
    if st.session_state.fail:
        raise RuntimeError('unavailable')
    rows = st.session_state.stores[source]
    if changes:
        patched = _v10_patch_rows(rows, changes, expected)
        st.session_state.writes.append((source, changes.copy()))
        st.session_state.stores[source] = patched
        return patched
    return rows.copy()
v10_load_settings_into_state()
if note := st.session_state.pop('_v10_notice', None):
    getattr(st, note[0])(note[1])
for key in _V10_RESTORE_KEYS:
    st.checkbox(key, key=key, on_change=v10_save_settings, args=(key,),
                disabled=not _v10_settings_allow_automation())
st.button('restore', on_click=v10_restore_risk_controls, disabled=not st.session_state._v10_settings_read_ok)
"""
        ).run()
        self.assertFalse(at.exception)
        return at

    def test_restore_only_three_and_no_write_on_rerun(self):
        other = [("v9_min_ev", "8.5"), ("v11_trial", "False"), ("LS|測試", "20")]
        at = self.boot([*( (k, "False") for k in KEYS), *other])
        self.assertFalse(any(at.checkbox(key=k).value for k in KEYS))
        self.assertFalse(at.session_state.writes)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(all(at.checkbox(key=k).value for k in KEYS))
        self.assertEqual(at.session_state.writes, [("Supabase", dict.fromkeys(KEYS, "True"))])
        self.assertEqual(dict(at.session_state.stores["Supabase"]), dict(other + [(k, "True") for k in KEYS]))
        at.run()
        reloaded = self.boot(at.session_state.stores["Supabase"])
        self.assertTrue(all(reloaded.checkbox(key=k).value for k in KEYS))
        self.assertFalse(reloaded.session_state.writes)
        for key in KEYS:
            at.checkbox(key=key).uncheck().run()
            self.assertFalse(at.checkbox(key=key).value)
            self.assertEqual(at.session_state.writes[-1], ("Supabase", {key: "False"}))
            at.button[0].click().run()
            self.assertEqual(at.session_state.writes[-1], ("Supabase", {key: "True"}))

    def test_conflict_on_any_of_three_aborts_entire_repair(self):
        at = self.boot([(k, "False") for k in KEYS])
        at.session_state.stores["Supabase"][-1] = (KEYS[-1], "True")
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertFalse(at.session_state.writes)
        self.assertTrue(at.warning)
        self.assertTrue(at.checkbox(key=KEYS[-1]).value)
        self.assertFalse(at.checkbox(key=KEYS[0]).value)

    def test_duplicates_repaired_without_overwriting_other_invalid_setting(self):
        at = self.boot([(KEYS[0], "True"), (KEYS[0], "False"), ("v9_min_ev", "broken")])
        self.assertTrue(at.checkbox(key=KEYS[0]).disabled)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.stores["Supabase"].count((KEYS[0], "True")), 1)
        self.assertIn(("v9_min_ev", "broken"), at.session_state.stores["Supabase"])
        self.assertTrue(at.checkbox(key=KEYS[0]).disabled)

    def test_fallback_keeps_last_read_until_source_changes(self):
        at = self.boot([(KEYS[0], "False"), ("v9_min_ev", "broken")])
        at.session_state.fail = True
        at.run()
        self.assertFalse(at.checkbox(key=KEYS[0]).value)
        self.assertTrue(at.button[0].disabled)
        at.session_state.source = "GitHub"
        at.session_state.fail = False
        at.run()
        self.assertTrue(at.checkbox(key=KEYS[0]).value)
        self.assertEqual(at.session_state.reads[-1], "GitHub")
        rounds = len(at.session_state.reads)
        at.run()
        self.assertEqual(len(at.session_state.reads), rounds)
        self.assertFalse(at.session_state.writes)


def ah(*, mid=999001, order=0, prob=.60, p_mkt=None, line=-.5, **patch):
    data = locked()
    data["odds"][0].update(type="讓球", line=line)
    data["odds"][0]["_clv_tipsme_hkjc_v1"][0] = str(mid)
    data["token"] = ("sid", 0, str(mid))
    data["fill"]["info"]["match_id"] = str(mid)
    time = ENTRY_AT + timedelta(seconds=order)
    row = base(ID=f"B{mid}{order}", Tipsme_Match_ID=mid, Date=time.isoformat(), Bet_Type="讓球",
               Selection="Away", Initial_Line=line, Gate="Y", Edge=.1, Unit_Profit=.95, **patch)
    row["Note"] = entry_note(row, CANDIDATE, data, KICK.timestamp(), time)
    assert row["Note"]
    market = p_mkt if p_mkt is not None else 1.85 / (1.85 + 1.95)
    record_handicap_forecast(row, {"p_model": prob, "p_mkt": market, "edge": prob-market})
    return row


class HandicapAudit(unittest.TestCase):
    def test_no_proof_vs_proven_zero(self):
        old = base(Bet_Type="讓球", Gate="N", Edge=-.02)
        self.assertEqual(handicap_audit([])["rows"], 0)
        audit = handicap_audit([old, ah(p_mkt=.55)])
        self.assertEqual((audit["rows"], audit["verified"], audit["forecasts"]), (2, 1, 0))
        audit = handicap_audit([ah(prob=.48)])
        self.assertEqual((audit["forecasts"], audit["passes"], audit["months"][0]["過閘等額回報%"]),
                         (1, 0, None))

    def test_earliest_per_fixture_binary_only_uses_frozen_values(self):
        first = ah(prob=.48)
        first.update(Edge=.8, Gate="Y", Market_Prob=.01, Model_Prob=.90)
        other = ah(mid=999002, line=-.25, System_Stake=10)
        other.update(Unit_Profit=-.5)
        audit = handicap_audit([ah(order=30), first, other, ah(mid=999003, Status="Open")])
        self.assertEqual((audit["verified"], audit["forecasts"], audit["passes"], audit["staked_passes"]),
                         (4, 3, 2, 1))
        month = audit["months"][0]
        self.assertEqual(month["概率評分場數"], 1)
        self.assertAlmostEqual(month["模型 Brier"], .52 ** 2, places=5)
        self.assertEqual(month["全部等額回報%"], 22.5)
        self.assertEqual(month["過閘等額回報%"], -50.0)


if __name__ == "__main__":
    unittest.main()
