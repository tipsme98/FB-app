"""Offline aggregate-only replay; no bet rows, account records or API calls are exported."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

import fb_app_clv_cohorts as clv
import fb_app_parameter_lab as lab


def emit(tag, payload):
    print(tag, json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def previous_function(path):
    functions = {n.name: n for n in ast.parse(path.read_text(encoding="utf-8-sig")).body
                 if isinstance(n, ast.FunctionDef)}
    needed = set()

    def visit(name):
        if name in needed or name not in functions:
            return
        needed.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Name):
                visit(node.id)

    visit("compute_bet_clv")
    scope = dict(pd=pd, np=np, json=json, re=re, datetime=datetime, timedelta=timedelta,
                 timezone=timezone, HKT=clv.HKT)
    exec(compile(ast.Module(body=[functions[n] for n in needed], type_ignores=[]), str(path), "exec"), scope)
    if "classify_clv" in scope["compute_bet_clv"].__code__.co_names:
        raise ValueError("Supply the original CLV implementation, not the candidate")
    return scope["compute_bet_clv"]


def replay(raw, previous):
    settled = raw.loc[raw.Status.eq("Settled")].copy()
    settled["old"] = pd.to_numeric(settled.apply(previous, axis=1), errors="coerce")
    settled["new"] = settled.apply(lambda row: clv.classify(row).group, axis=1)
    logged = pd.to_datetime(settled.Date, format="mixed", errors="coerce").dt.normalize()
    day = pd.to_datetime(settled.Match_Date, format="mixed", errors="coerce").dt.normalize()
    settled["date_evidence"] = np.select(
        [logged.gt(day), logged.eq(day), logged.lt(day)], ["later_day", "same_day", "earlier_day"],
        default="missing")
    archived = pd.to_numeric(settled["CLV%"], errors="coerce") if "CLV%" in settled else None
    emit("COVERAGE", dict(rows=len(raw), statuses=raw.Status.value_counts().to_dict(),
                          cohorts=settled.new.value_counts().to_dict(),
                          archived_n=int(archived.notna().sum()) if archived is not None else None,
                          matching_at_export_precision=int(archived.eq((settled.old * 100).round(2)).sum())
                          if archived is not None else None))
    for tag, g in [("all", settled), *settled.groupby("Bet_Type"), *settled.groupby("date_evidence")]:
        x = g.old.dropna()
        emit("OLD", dict(group=tag, rows=len(g), compared=len(x), exact_zero=int(x.eq(0).sum()),
                         within_0_01_percentage_point=int(x.abs().le(.0001).sum()),
                         mean_percent=x.mean()*100 if len(x) else None,
                         same_day=int(g.date_evidence.eq("same_day").sum())))


def fixed_shadow(raw):
    """Pre-specified intercept shrinkage; no feature or threshold selection on holdout."""
    data, _, coverage = lab.get_dataset(raw)
    emit("SNAPSHOT_PROXY", coverage)
    split = pd.Timestamp("2026-10-17")
    for market in lab.MARKETS:
        d = data[data.market.eq(market)].sort_values(["date", "key"])
        # Exclude the full preceding Hong Kong date to avoid uncertain result availability.
        train, test = d[d.date < split - lab.DAY], d[d.date >= split]
        if train.empty or test.empty:
            continue
        anchor = logit(train.p.to_numpy())
        w, total = train.w.to_numpy(), (train.w + train.l).to_numpy()
        # 25% of the regularized intercept on market log-odds.
        bias = brentq(lambda x: np.sum(total * expit(anchor + x) - w) + 20*x, -8, 8) / 4
        base = test.p.to_numpy()
        prediction = expit(logit(base) + bias)
        w, l = test.w.to_numpy(), test.l.to_numpy()
        market_loss = -w*np.log(base) - l*np.log1p(-base)
        delta = -w*np.log(prediction) - l*np.log1p(-prediction) - market_loss
        # +2% estimated EV threshold and sensitivity charging 0.5% of each stake.
        action = float(total.mean())
        hi, ev = lab._sides(test, prediction, np.full(len(test), action))
        chosen = ev >= .02
        returns = np.where(hi, test.rhi, test.rlo)
        dated = pd.DataFrame(dict(day=test.date.to_numpy(), delta=delta, weight=w+l,
                                  roi=np.where(chosen, returns-.005, 0), chosen=chosen.astype(int)))
        agg = dated.groupby("day").sum().to_numpy()
        rng = np.random.default_rng(23117)
        draws = agg[rng.integers(0, len(agg), size=(4000, len(agg)))].sum(axis=1)
        loss_ci = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
        nz = draws[:, 3] > 0
        roi_ci = np.quantile(draws[nz, 2] / draws[nz, 3], [.025, .975]).tolist() if nz.any() else None
        emit("FIXED_SHADOW", dict(
            market=market, train=len(train), test=len(test), first=test.date.min().date(),
            last=test.date.max().date(), market_log_loss=float(market_loss.sum()/(w+l).sum()),
            paired_loss_change=float(delta.sum()/(w+l).sum()), change_ci=loss_ci,
            orders=int(chosen.sum()), roi=float(returns[chosen].mean()) if chosen.any() else None,
            roi_with_cost=float(returns[chosen].mean()-.005) if chosen.any() else None,
            roi_with_cost_ci=roi_ci, applied_offset=bias))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workbook", type=Path)
    p.add_argument("--old-app", type=Path, required=True)
    p.add_argument("--ledger-only", action="store_true")
    args = p.parse_args()
    raw = pd.read_excel(args.workbook, sheet_name="Sheet1", dtype=str, keep_default_na=False)
    replay(raw, previous_function(args.old_app))
    if not args.ledger_only:
        fixed_shadow(raw)


if __name__ == "__main__":
    main()
