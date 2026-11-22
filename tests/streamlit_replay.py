"""Test-only Streamlit entry. Synthetic network inputs, clock and history; no Secrets."""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pandas as pd
import requests
import streamlit as st

import fb_app_t30_shadow as shadow
from test_t30_shadow import KO, CUT, HOME, feed, profile_data


def sample_history():
    history = []
    for i in range(80):
        day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=i)
        stamped = (day - pd.Timedelta(hours=3)).isoformat() + "+08:00"
        odds = [dict(type=m, line=line, upper=1.90, lower=1.93, record_time=stamped, unlock=True)
                for m, line in zip(shadow.MARKETS, (-.25, 2.25, 9.5))]
        history.append(dict(Status="Settled", Match_Date=day.isoformat(), Home_Team=f"主{i % 5}",
                            Away_Team=f"客{i % 5}", Home_Goal=(i % 3) * 2, Away_Goal=i % 2,
                            Home_Corner=3 + (i % 5) * 2, Away_Corner=3 + i % 2, Odds_History=json.dumps(odds),
                            Tournament_Name=HOME["league"], Tipsme_Match_ID=str(1000 + i)))
    return pd.DataFrame(history)


def run():
    if os.environ.get("T30_TEST_REPLAY") != "synthetic-only":
        raise RuntimeError("This entry is for isolated synthetic tests only.")
    st.warning("離線合成測試｜非實時賽事或報價")
    if "_replay_time" not in st.session_state:
        st.session_state["_replay_time"] = CUT if os.environ.get("T30_TEST_PHASE") == "cutoff" else CUT - 75
    if st.button("測試專用：前進至 T−30"):
        st.session_state["_replay_time"] = CUT

    app = runpy.run_path(str(Path(__file__).resolve().parents[1] / "FB-app.py"),
                         run_name="fb_app_streamlit_replay")["main"].__globals__
    if any(app["_has_secret"](k) for k in ("DB_URL", "GITHUB_TOKEN", "GITHUB_REPO")):
        raise RuntimeError("Synthetic replay must never run with live Secrets.")

    def clock():
        return datetime.fromtimestamp(st.session_state["_replay_time"], shadow.HKT)

    def fake_schedule(url, **kwargs):
        if kwargs.get("marker") != "matchSchedules":
            return None
        if url != "https://tipsme.hk/zh-HK":
            raise AssertionError("Use the canonical schedule URL that avoids the trailing-slash redirect")
        event = dict(matchId=42, home=HOME["home"], away=HOME["away"], leagueName=HOME["league"],
                     fullMatchTime=shadow._iso(KO), matchStatus="fixture-future", isHkjc=True)
        raw = dict(props=dict(pageProps=dict(matchSchedules=[event])))
        return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(raw)}</script>'

    def fake_get(url, **kwargs):
        if not url.endswith("/Score/odds/hkjc/42") or clock().timestamp() >= CUT:
            raise AssertionError(f"Unexpected request in replay: {url}")
        data = feed((clock() - timedelta(seconds=2)).timestamp())
        scenario = os.environ.get("T30_TEST_VARIANT")
        if scenario == "rejected":
            data["book"]["hdp"]["state"] = "CLOSED"
            del data["book"]["hilo"]["l"]
            data["book"]["corners"]["stamp"] = shadow._iso(CUT - shadow.WINDOW - 1)
        elif scenario == "mismatch":
            data["fixture"] = 43
        return SimpleNamespace(status_code=200, text=json.dumps(data))

    def denied_write(*args, **kwargs):
        raise AssertionError("Replay reached a regular ledger write")

    app.update(get_hkt_now=clock, _tipsme_fetch_page_html=fake_schedule, _tipsme_get=lambda url: None,
               requests=SimpleNamespace(get=fake_get, RequestException=requests.RequestException), save_db=denied_write)
    orig_tick = app["_t30_shadow_tick"]

    def replay_tick():
        history = st.session_state.get("df_db")
        try:
            if os.environ.get("T30_TEST_VARIANT") != "no-history":
                st.session_state["df_db"] = sample_history()
            return orig_tick()
        finally:
            st.session_state["df_db"] = history

    app["_t30_shadow_tick"] = replay_tick
    app["main"]()


if __name__ == "__main__":
    run()
