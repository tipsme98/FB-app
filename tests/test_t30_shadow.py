import ast
from copy import deepcopy
from contextlib import closing
import csv
from datetime import datetime, timedelta
import json
import io
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fb_app_t30_shadow as shadow

KO = datetime.fromisoformat("2026-11-23T00:10:00+08:00").timestamp()
CUT = KO - shadow.CUTOFF
HOME = {"hkjc_match_id": 42, "home": "主隊", "away": "客隊", "league": "聯賽", "match_time": KO,
        "shadow_status": "fixture-future", "shadow_is_hkjc": True}


def profile_data():
    return dict(evidence="SYNTHETIC ONLY; not a Tipsme data contract", schedule_prematch=["fixture-future"],
                fixture_id="fixture", handicap_sign=1, naive_zone=None, markets={
                    m: dict(path=f"book.{code}", status="state", timestamp="stamp", line="line",
                            upper="u", lower="l", open=["OPEN"], closed=["CLOSED"])
                    for m, code in zip(shadow.MARKETS, ("hdp", "hilo", "corners"))})


def feed(ts):
    return dict(fixture=42, book={m: dict(state="OPEN", stamp=shadow._iso(ts), line=line, u=1.92, l=1.91)
                                  for m, line in zip(("hdp", "hilo", "corners"), (-.25, 2.25, 9.5))})


class ShadowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = shadow.Store(self.temp.name)
        self.profile = shadow.Profile(profile_data())
        self.calls = []

    def schedule(self, t=CUT-90, row=HOME):
        self.store.ingest_schedule([row], t)

    def capture(self, t=CUT-60, source=None, raw=None):
        return self.store.capture(42, feed(t if source is None else source) if raw is None else raw, t, self.profile)

    def score(self, fixture, now):
        self.calls.append((fixture, now))
        assert fixture["kickoff"] == shadow._iso(KO)
        output = {}
        for q in fixture["odds"]:
            output[q["type"]] = {"盤口": q["type"], "主盤": q["line"], "主或大賠率": q["upper"],
                                 "客或小賠率": q["lower"], "報價時間": q["record_time"],
                                 "預測方向": "主" if q["type"] == "讓球" else "大",
                                 "建議": "略過" if q["type"] == "讓球" else "影子模擬"}
        return list(output.values()), {"training_input_sha256": "test"}

    def due(self, t=CUT, profile=True, scorer=None):
        return self.store.run_due(lambda: t, self.profile if profile else None, scorer or self.score)

    def rows(self):
        return {r["market"]: r for r in self.store.rows()}

    def test_three_markets_and_idempotency_across_instances(self):
        self.schedule()
        self.assertTrue(self.capture())
        self.assertEqual(self.due(CUT-1), 0)
        self.assertEqual(self.due(), 1)
        self.store = shadow.Store(self.temp.name)
        self.assertEqual(self.due(CUT+1), 0)
        self.assertEqual(self.rows()["讓球"]["reason"], "forecast_no_pick")
        self.assertEqual(self.rows()["角球大小"]["timing"], "at_cutoff")
        self.assertEqual(self.rows()["入球大小"]["kickoff"], "2026-11-23T00:10:00+08:00")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0][0]["odds"]), 3)

    def test_unverified_profile_no_extra_fetch_no_scoring(self):
        self.schedule()
        self.assertIsNone(self.store.next_fetch(CUT-50, None))
        self.due(profile=False)
        self.assertFalse(self.calls)
        self.assertEqual({x["reason"] for x in self.rows().values()}, {"source_unverified"})

    def test_missing_and_newly_closed_do_not_block_other_market_or_rewind(self):
        self.schedule()
        self.capture()
        event = feed(CUT-30)
        event["book"]["hilo"]["state"] = "CLOSED"
        del event["book"]["corners"]["l"]
        self.capture(CUT-30, raw=event)
        self.due(CUT+18)
        rows = self.rows()
        self.assertEqual(rows["入球大小"]["reason"], "market_closed")
        self.assertEqual(rows["角球大小"]["reason"], "invalid_two_sided_quote")
        self.assertEqual(rows["讓球"]["decision_lag_s"], 18)
        self.assertEqual(rows["讓球"]["timing"], "delayed")
        self.assertEqual({q["type"] for q in self.calls[0][0]["odds"]}, {"讓球"})
        self.assertEqual(len(self.calls[0][0]["odds"]), 2)  # Both valid time points retain drift.

    def test_future_observation_and_future_source(self):
        self.schedule()
        self.assertFalse(self.capture(CUT+1, CUT-10))
        self.capture(CUT-20, CUT-10)
        self.due()
        self.assertEqual({x["reason"] for x in self.rows().values()}, {"quote_after_collection"})

    def test_quote_window_endpoints_and_utc(self):
        self.schedule(CUT-900)
        event = feed(CUT-900)
        for q in event["book"].values():
            q["stamp"] = datetime.fromtimestamp(CUT-900, shadow.timezone.utc).isoformat().replace("+00:00", "Z")
        self.assertTrue(self.capture(CUT-900, raw=event))
        self.schedule(CUT-5)
        self.assertTrue(self.capture(CUT, CUT))
        self.due()
        self.assertEqual(len(self.calls[0][0]["odds"]), 6)
        self.assertEqual(self.rows()["讓球"]["collected_at"], shadow._iso(CUT))

    def test_no_zone_without_reviewed_contract(self):
        self.schedule()
        event = feed(CUT-30)
        event["book"]["hdp"]["stamp"] = "2026-11-22T23:39:30"
        self.capture(CUT-30, raw=event)
        self.due()
        self.assertEqual(self.rows()["讓球"]["reason"], "quote_time_unknown")
        self.assertEqual(self.rows()["角球大小"]["reason"], "forecast_pick")

    def test_quote_fixture_mismatch(self):
        self.schedule()
        self.capture(raw=dict(feed(CUT-60), fixture=99999))
        self.due()
        self.assertEqual({r["reason"] for r in self.rows().values()}, {"quote_fixture_mismatch"})

    def test_open_enum_not_inferred_from_boolean_or_historic_list(self):
        self.schedule()
        event = feed(CUT-60)
        event["book"]["hilo"]["state"] = True
        event["book"]["corners"] = [event["book"]["corners"]]
        self.capture(raw=event)
        self.due()
        self.assertEqual(self.rows()["入球大小"]["reason"], "market_status_unknown")
        self.assertEqual(self.rows()["角球大小"]["reason"], "market_missing")

    def test_source_same_time_price_conflict(self):
        self.schedule()
        self.capture()
        event = feed(CUT-60)
        event["book"]["hilo"]["u"] = 1.95
        self.capture(CUT-30, raw=event)
        self.due()
        self.assertEqual(self.rows()["入球大小"]["reason"], "quote_conflict_or_time_regression")

    def test_unverified_match_status_never_fetches(self):
        self.schedule(row={**HOME, "shadow_status": None})
        self.assertIsNone(self.store.next_fetch(CUT-60, self.profile))
        self.assertFalse(self.capture())
        self.due()
        self.assertEqual(self.rows()["讓球"]["reason"], "fixture_not_verified_prematch")

    def test_refresh_error_revokes_schedule(self):
        self.schedule()
        self.capture()
        self.store.ingest_schedule([], CUT-15)
        self.due()
        self.assertEqual(self.rows()["讓球"]["reason"], "schedule_not_fresh")

    def test_reschedule_quarantines_previous_kickoff(self):
        self.schedule()
        self.capture()
        self.schedule(CUT-15, {**HOME, "match_time": KO+3600})
        self.due()
        self.assertEqual(self.rows()["讓球"]["reason"], "rescheduled")
        self.assertFalse(self.calls)

    def test_competing_kickoffs_are_ambiguous(self):
        self.store.ingest_schedule([HOME, {**HOME, "match_time": KO+360}], CUT-50)
        self.assertIsNone(self.store.next_fetch(CUT-30, self.profile))
        self.due()
        self.assertEqual(self.rows()["讓球"]["reason"], "schedule_conflict")

    def test_one_schedule_record_with_conflicting_time_sources_is_quarantined(self):
        self.schedule(row={**HOME, "shadow_kickoff_conflict": True})
        self.assertFalse(self.capture())
        self.due()
        self.assertEqual(self.rows()["讓球"]["reason"], "schedule_conflict")

    def test_late_engine_result_is_not_a_prediction(self):
        self.schedule()
        self.capture()
        clocks = iter([CUT, CUT, CUT, CUT+151])
        self.store.run_due(lambda: next(clocks), self.profile, self.score)
        self.assertEqual(self.rows()["入球大小"]["reason"], "decision_window_missed")
        self.assertIsNone(self.rows()["入球大小"]["strategy_units"])
        self.assertNotIn("forecast_建議", self.rows()["入球大小"])

    def test_after_kickoff_does_not_replay(self):
        self.schedule()
        self.capture()
        self.due(KO+1)
        self.assertEqual(self.rows()["讓球"]["reason"], "decision_window_missed")
        self.assertFalse(self.calls)

    def test_reuse_and_fetch_throttle_across_store_instances(self):
        self.schedule(CUT-240)
        self.capture(CUT-200)
        self.assertIsNone(self.store.next_fetch(CUT-160, self.profile))
        self.assertEqual(self.store.next_fetch(CUT-115, self.profile), "42")
        self.assertIsNone(shadow.Store(self.temp.name).next_fetch(CUT-100, self.profile))
        self.assertIsNone(self.store.next_fetch(CUT, self.profile))

    def test_global_fetch_budget_across_multiple_runners(self):
        self.store.ingest_schedule([HOME, {**HOME, "hkjc_match_id": 43}], CUT-130)
        self.assertEqual(self.store.next_fetch(CUT-115, self.profile), "42")
        other = shadow.Store(self.temp.name)
        self.assertIsNone(other.next_fetch(CUT-100, self.profile))
        self.assertEqual(other.next_fetch(CUT-90, self.profile), "43")

    def test_result_labels_no_pick_missing_corners_and_conflict(self):
        self.schedule()
        self.capture()
        self.due()
        match = {**HOME, "finished": True, "home_goal": 1, "away_goal": 1,
                 "home_corner": None, "away_corner": None}
        self.assertEqual(self.store.ingest_results([{**match, "match_time": KO+1}], KO+7200), 0)
        self.assertEqual(self.store.ingest_results([{**match, "away": "錯配"}], KO+7200), 0)
        self.assertEqual(self.store.ingest_results([match], KO+7200), 2)
        rows = self.rows()
        self.assertEqual(rows["讓球"]["outcome"], "半輸")
        self.assertIsNone(rows["讓球"]["strategy_units"])
        self.assertEqual(rows["入球大小"]["strategy_units"], -.5)
        self.assertEqual(rows["角球大小"]["result_state"], "pending")
        self.assertEqual(self.store.ingest_results([{**match, "home_corner": 7, "away_corner": 3}], KO+7201), 1)
        self.assertEqual(self.rows()["角球大小"]["outcome"], "全贏")
        self.store.ingest_results([{**match, "home_goal": 2}], KO+7202)
        self.assertEqual(self.rows()["入球大小"]["result_state"], "conflict")
        self.assertIsNone(self.rows()["入球大小"]["strategy_units"])

    def test_all_five_asian_outcomes_and_lower_odds(self):
        q = {"line": 2.75, "upper": 1.9, "lower": 1.8}
        samples = [(2.5, 3, "全贏", .9), (2.75, 3, "半贏", .45), (3, 3, "走盤", 0),
                   (3.25, 3, "半輸", -.5), (3.5, 3, "全輸", -1)]
        for line, goals, label, units in samples:
            actual = shadow.settlement("入球大小", {**q, "line": line}, "大", goals, 0)
            self.assertEqual(actual, {"outcome": label, "hypothetical_units": units})
        self.assertEqual(shadow.settlement("入球大小", q, "小", 2, 0)["hypothetical_units"], .8)

    def test_no_touch_of_unrelated_database(self):
        with tempfile.TemporaryDirectory() as folder:
            unrelated = Path(folder) / "fb-app-t30-shadow.sqlite3"
            with closing(sqlite3.connect(unrelated)) as conn:
                conn.execute("create table football_bets (id integer)")
                conn.commit()
            before = unrelated.read_bytes()
            with self.assertRaisesRegex(ValueError, "Not an FB-app"):
                shadow.Store(folder)
            self.assertEqual(unrelated.read_bytes(), before)

    def test_profile_contract_cannot_use_the_same_status_for_open_and_closed(self):
        raw = deepcopy(profile_data())
        raw["markets"]["入球大小"]["closed"] = ["OPEN"]
        with self.assertRaises(ValueError):
            shadow.Profile(raw)

    def test_profile_requires_explicit_quote_fixture_identity(self):
        raw = profile_data()
        del raw["fixture_id"]
        with self.assertRaises(ValueError):
            shadow.Profile(raw)

    def test_reference_forecast_and_legacy_reference_never_add_strategy_units(self):
        def score(fixture, now):
            output, provenance = self.score(fixture, now)
            for row in output:
                if row["盤口"] == "角球大小":
                    row["建議"] = "對照訊號"
            return output, provenance
        self.schedule()
        self.capture()
        self.due(scorer=score)
        self.store.ingest_results([{**HOME, "finished": True, "home_corner": 7, "away_corner": 3}], KO+7200)
        ref = self.rows()["角球大小"]
        self.assertEqual(ref["reason"], "forecast_reference")
        self.assertEqual((ref["hypothetical_units"], ref["strategy_units"]), (.92, None))
        # RC1 used forecast_pick for both recommendation types. Verify read compatibility.
        with self.store._db() as conn:
            conn.execute("UPDATE decisions SET reason='forecast_pick' WHERE market=?", ("角球大小",))
        self.assertIsNone(self.rows()["角球大小"]["strategy_units"])

    def test_csv_escapes_whitespace_prefixed_formulas_and_preserves_numbers(self):
        values = dict(home="  =1+1", away="\nhello", league="普通", line=-.25)
        exported = shadow.decisions_csv([values])
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(next(csv.DictReader(io.StringIO(exported.decode("utf-8-sig")))),
                         dict(home="'  =1+1", away="'\nhello", league="普通", line="-0.25"))

    def test_schedule_health_and_read_only_existing_store(self):
        r = shadow.Store(self.temp.name, read_only=True)
        self.assertEqual(r.schedule_health(CUT), ("waiting", None))
        self.schedule()
        self.assertEqual(r.schedule_health(CUT), ("loaded", shadow._iso(CUT-90)))
        self.assertEqual(r.schedule_health(CUT+400), ("unavailable", shadow._iso(CUT-90)))
        with self.assertRaises(sqlite3.OperationalError):
            r.reserve("should-not-write", CUT, 30)

    def test_export_separates_no_pick_and_result(self):
        self.schedule()
        self.capture()
        self.due()
        path = Path(self.temp.name) / "result.csv"
        with patch("sys.argv", ["prog", self.temp.name, str(path)]):
            shadow.main()
            with self.assertRaises(FileExistsError):
                shadow.main()
        content = path.read_text(encoding="utf-8-sig")
        self.assertIn("forecast_no_pick", content)
        self.assertIn("forecast_pick", content)
        self.assertIn("hypothetical_units", content)

    def test_real_engine_with_synthetic_history(self):
        import pandas as pd
        import fb_app_parameter_lab as lab
        history = []
        for i in range(80):
            day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=i)
            stamped = (day - pd.Timedelta(hours=3)).isoformat() + "+08:00"
            odds = [dict(type=m, line=line, upper=1.90, lower=1.93, record_time=stamped, unlock=True)
                    for m, line in zip(shadow.MARKETS, (-.25, 2.25, 9.5))]
            history.append(dict(Status="Settled", Match_Date=day.isoformat(), Home_Team=f"主{i%5}",
                                Away_Team=f"客{i%5}", Home_Goal=(i%3)*2, Away_Goal=i%2,
                                Home_Corner=3+(i%5)*2, Away_Corner=3+i%2, Odds_History=json.dumps(odds),
                                Tournament_Name=HOME["league"], Tipsme_Match_ID=str(1000+i)))
        raw = pd.DataFrame(history)
        self.schedule()
        self.capture(source=CUT-60.125, t=CUT-60)
        def real(fixture, now):
            return lab.score_fixture(raw, fixture, now=pd.Timestamp(now.replace(tzinfo=None))), {"engine": lab.LAB_VERSION}
        self.due(scorer=real)
        self.assertTrue(all("forecast_模型有效比例" in r for r in self.rows().values()), self.rows())
        self.assertEqual(self.rows()["角球大小"]["provenance_engine"], lab.LAB_VERSION)


class AppWiringTest(unittest.TestCase):
    SRC = ast.parse((Path(__file__).resolve().parents[1] / "FB-app.py").read_text())

    def env(self, *names, **scope):
        funcs = [n for n in self.SRC.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(funcs, []), "app-wiring", "exec"), scope)
        return scope

    def test_feature_defaults_off_and_viewer_never_touches_store(self):
        def fail():
            raise AssertionError("No access to shadow storage expected")
        env = self.env("_t30_shadow_enabled", "_t30_shadow_relay",
                       os=os, _v162_is_runner=lambda: False, _t30_shadow_context=fail)
        with patch.dict(os.environ, {"FB_APP_SHADOW_MODE": "observe"}):
            env["_t30_shadow_relay"]("schedule", [HOME])
        env["_v162_is_runner"] = lambda: True
        with patch.dict(os.environ, {"FB_APP_SHADOW_MODE": ""}):
            env["_t30_shadow_relay"]("schedule", [HOME])

    def test_existing_schedule_fallback_not_relabelled_fresh(self):
        calls = []
        clock = datetime.fromtimestamp(CUT, shadow.HKT)
        state = {"_v16_sched_cache": {"ts": CUT-600, "matches": [HOME]}}
        env = self.env("_v16_today_cache", get_hkt_now=lambda: clock, st=SimpleNamespace(session_state=state),
                       tipsme_fetch_today_schedule=lambda: [], _t30_shadow_relay=lambda *v: calls.append(v))
        self.assertEqual(env["_v16_today_cache"](), [HOME])
        self.assertEqual(calls, [("schedule", [])])

    def test_one_existing_odds_response_can_relay_without_changing_return(self):
        calls = []
        raw = feed(CUT-60)
        env = self.env("tipsme_fetch_hkjc_odds", json=json, TIPSME_API_BASE="https://test.invalid",
                       _tipsme_get=lambda url: json.dumps(raw), _t30_shadow_relay=lambda *v: calls.append(v))
        self.assertEqual(env["tipsme_fetch_hkjc_odds"](42), raw)
        self.assertEqual(calls, [("odds", raw, 42)])

    def test_shadow_store_failure_does_not_modify_production_response(self):
        def broken(*args):
            raise OSError
        state = {}
        env = self.env("tipsme_fetch_hkjc_odds", "_t30_shadow_relay", json=json, TIPSME_API_BASE="https://test.invalid",
                       _tipsme_get=lambda url: '{"test":true}', _t30_shadow_context=broken,
                       _t30_shadow_enabled=lambda: True, _v15_is_control_exc=lambda e: False,
                       st=SimpleNamespace(session_state=state), get_hkt_now=lambda: datetime.fromtimestamp(CUT, shadow.HKT))
        self.assertEqual(env["tipsme_fetch_hkjc_odds"](42), {"test": True})
        self.assertEqual(state["_t30_shadow_last_error"], "OSError")

    def test_runner_tick_writes_three_forecasts_with_provenance_and_no_extra_fetch(self):
        import pandas as pd
        import fb_app_parameter_lab as lab
        with tempfile.TemporaryDirectory() as folder:
            store, profile = shadow.Store(folder), shadow.Profile(profile_data())
            store.ingest_schedule([HOME], CUT-100)
            store.capture(42, feed(CUT-40), CUT-40, profile)
            clock = datetime.fromtimestamp(CUT, shadow.HKT)
            state = {"df_db": pd.DataFrame([
                dict(Status=status, Match_Date="2026-11-01", Home_Team="h", Away_Team="a", Home_Goal=0,
                     Away_Goal=0, Odds_History="[]", System_Stake=500)
                for status in ("Settled", "Open")
            ])}
            def raw_fetch(*args, **kwargs):
                self.fail("Runner must not fetch another odds snapshot after the cutoff")
            env = self.env("_t30_shadow_tick", pd=pd, get_hkt_now=lambda: clock,
                           st=SimpleNamespace(session_state=state), _t30_shadow_context=lambda: (store, profile),
                           _v16_today_cache=lambda: [HOME], tipsme_fetch_hkjc_odds=raw_fetch)
            def runner_score(history, fx, now):
                self.assertEqual(list(history["Status"]), ["Settled"])
                self.assertNotIn("System_Stake", history)
                self.assertEqual(now, pd.Timestamp("2026-11-22T23:40:00"))
                return [{"盤口": q["type"], "主盤": q["line"], "主或大賠率": q["upper"], "客或小賠率": q["lower"],
                         "報價時間": q["record_time"], "預測方向": "主" if q["type"] == "讓球" else "大",
                         "建議": "略過"} for q in fx["odds"]]
            with patch.object(lab, "score_fixture", runner_score):
                self.assertEqual(env["_t30_shadow_tick"](), (1, True))
                self.assertEqual(env["_t30_shadow_tick"](), (0, True))
            rows = list(store.rows())
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(len(r["provenance_training_input_sha256"]) == 64 for r in rows))
            self.assertTrue(all(len(r["provenance_engine_sha256"]) == 64 for r in rows))


if __name__ == "__main__":
    unittest.main()
