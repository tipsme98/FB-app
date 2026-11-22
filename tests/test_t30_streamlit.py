"""Run the real FB-app page via Streamlit AppTest using isolated synthetic inputs."""

import csv
from contextlib import closing
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import fb_app_t30_shadow as shadow
from test_t30_shadow import profile_data

REPLAY = Path(__file__).with_name("streamlit_replay.py")


class StreamlitReplayTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.profile = self.root / "synthetic-profile.json"
        self.profile.write_text(json.dumps(profile_data()), encoding="utf-8")
        self.store = self.root / "shadow"
        self.store.mkdir(mode=0o700)
        origin = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, origin)

    def env(self, variant="", mode="observe", profile=True, phase=""):
        return patch.dict(os.environ, dict(T30_TEST_REPLAY="synthetic-only", T30_TEST_VARIANT=variant,
                                          T30_TEST_PHASE=phase, FB_APP_SHADOW_MODE=mode,
                                          FB_APP_SHADOW_DIR=str(self.store),
                                          FB_APP_SHADOW_PROFILE=str(self.profile) if profile else ""))

    def app(self, runner=True):
        at = AppTest.from_file(str(REPLAY), default_timeout=15)
        if runner:
            at.query_params["runner"] = "1"
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.tabs), 7)
        return at

    def at_cutoff(self, at):
        next(b for b in at.button if b.label == "測試專用：前進至 T−30").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertFalse(any("未能讀取影子" in w.value or "影子本次未能執行" in w.value for w in at.warning))

    def rows(self):
        return list(shadow.Store(self.store).rows(recent=100))

    def test_disabled_runner_and_enabled_viewer_never_create_shadow_store(self):
        with self.env(mode=""):
            self.assertFalse(any("T−30 三盤本地影子" in e.label for e in self.app().expander))
        with self.env():
            self.assertFalse(any("T−30 三盤本地影子" in e.label for e in self.app(runner=False).expander))
        self.assertEqual(list(self.store.iterdir()), [])

    def test_accept_lock_export_reload_and_new_runner(self):
        with self.env():
            first = self.app()
            db = self.store / "fb-app-t30-shadow.sqlite3"
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(conn.execute("select count(*) from captures").fetchone()[0], 1)
                self.assertEqual(conn.execute("select count(*) from decisions").fetchone()[0], 0)
            self.at_cutoff(first)
            rows = self.rows()
            self.assertEqual({r["market"] for r in rows}, set(shadow.MARKETS))
            self.assertTrue(all(r.get("forecast_預測方向") and r.get("provenance_engine_sha256") for r in rows), rows)
            self.assertTrue(any(r["reason"] in ("forecast_no_pick", "forecast_reference") for r in rows))
            table = next(d.value for d in first.dataframe if "方向" in d.value.columns)
            self.assertEqual(len(table), 3)
            self.assertTrue(all(table["狀態"].str.startswith("預測：")))
            self.assertTrue(first.get("download_button"))
            export = list(csv.DictReader(io.StringIO(shadow.decisions_csv(rows).decode("utf-8-sig"))))
            self.assertEqual({r["market"] for r in export}, set(shadow.MARKETS))
            self.at_cutoff(first)
        with self.env(phase="cutoff"):
            self.app()
        self.assertEqual(self.rows(), rows)
        self.assertFalse(list(self.root.glob("football*")))

    def test_rejections_and_missing_history_in_ui(self):
        scenarios = (
            ("rejected", True, {"market_closed", "invalid_two_sided_quote", "quote_stale_or_after_cutoff"}),
            ("mismatch", True, {"quote_fixture_mismatch"}),
            ("no-history", True, {"training_or_model_unavailable"}),
            ("", False, {"source_unverified"}),
        )
        for i, (variant, configured, expected) in enumerate(scenarios):
            with self.subTest(variant=variant, configured=configured):
                self.store = self.root / f"shadow-{i}"
                self.store.mkdir(mode=0o700)
                with self.env(variant=variant, profile=configured):
                    at = self.app()
                    self.at_cutoff(at)
                rows = self.rows()
                self.assertEqual({r["reason"] for r in rows}, expected)
                table = next(d.value for d in at.dataframe if "方向" in d.value.columns)
                self.assertEqual(len(table), 3)
                self.assertTrue(all(table["狀態"].str.startswith("拒收：")))
                self.assertTrue(table["方向"].isna().all())


if __name__ == "__main__":
    unittest.main()
