"""Read-only comparison of submit-time handicap forecasts against the locked market."""

from collections import defaultdict
from math import isclose, log
from statistics import mean

from fb_app_clv_cohorts import ENTRY, number, report_period, verified_entry


def _probabilities(entry):
    key, _, _, _, prices, frozen = entry
    if not isinstance(frozen, list) or len(frozen) != 3 or frozen[0] != "V15.8_LR":
        return None
    p, logged = (number(x) for x in frozen[1:])
    if p is None or logged is None or not (0 < p < 1 and 0 < logged < 1):
        return None
    side = int(key[3] == "Away")
    fair = prices[1 - side] / sum(prices)
    # The legacy gate may otherwise compare against a different handicap line.
    if not isclose(fair, logged, abs_tol=1e-6, rel_tol=0):
        return None
    return p, fair, p - logged > 0.02


def _unit(row, price):
    unit = number(row.get("Unit_Profit"))
    if unit is None or row.get("Status") != "Settled":
        return None
    if not any(isclose(unit, v, rel_tol=0, abs_tol=.0051)
               for v in (-1, -.5, 0, (price - 1) / 2, price - 1)):
        return None
    return unit


def handicap_audit(rows):
    counts = dict(rows=0, logged_scores=0, logged_pass=0, verified=0, forecasts=0,
                  passes=0, staked_passes=0, months=[])
    choices = {}
    for row in rows:
        note = str(row.get("Note", "")).strip()
        if row.get("Bet_Type") != "讓球" or note == "用家跟投" or report_period(row.get("Date")) != 1:
            continue
        counts["rows"] += 1
        gate, edge = row.get("Gate"), number(row.get("Edge"))
        if gate in ("Y", "N") and edge is not None:
            counts["logged_scores"] += 1
            counts["logged_pass"] += gate == "Y"
        entry = verified_entry(row) if note.startswith(ENTRY) else None
        if entry is None:
            continue
        counts["verified"] += 1
        probabilities = _probabilities(entry)
        if probabilities is None:
            continue
        key, _, time, _, _, _ = entry
        order = (time, key[0])
        prior = choices.get(key[1])
        if prior is None or order < prior[0]:
            choices[key[1]] = (order, key, probabilities, row)
    months = defaultdict(lambda: ([], [], []))
    for order, key, (p, market, passed), row in choices.values():
        counts["forecasts"] += 1
        counts["passes"] += passed
        counts["staked_passes"] += bool(passed and (number(row.get("System_Stake")) or 0) > 0)
        unit = _unit(row, key[5])
        if unit is None:
            continue
        scored, all_returns, pass_returns = months[order[0].strftime("%Y-%m")]
        all_returns.append(unit)
        if passed:
            pass_returns.append(unit)
        # Only .5 fractional lines are binary; quarter lines pay half and integer lines can push.
        if abs(key[4]) % 1 == .5 and (isclose(unit, -1, abs_tol=.0051)
                                       or isclose(unit, key[5] - 1, abs_tol=.0051)):
            y = int(unit > 0)
            scored.append(((p - y) ** 2, (market - y) ** 2,
                           -log(p if y else 1 - p), -log(market if y else 1 - market)))
    for month, (scored, all_returns, pass_returns) in sorted(months.items()):
        scores = list(map(mean, zip(*scored))) if scored else [None] * 4
        counts["months"].append({
            "入注月（香港）": month, "概率評分場數": len(scored),
            **dict(zip(("模型 Brier", "市場 Brier", "模型對數損失", "市場對數損失"),
                       (round(s, 5) if s is not None else None for s in scores))),
            "全部結算場數": len(all_returns), "全部等額回報%": round(100 * mean(all_returns), 2),
            "過閘結算場數": len(pass_returns),
            "過閘等額回報%": round(100 * mean(pass_returns), 2) if pass_returns else None,
        })
    return counts
