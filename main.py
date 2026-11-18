"""Forecast a launch day's Twitch viewers, hour by hour.

Competition: gta6-twitch-ccu-forecast-01. The program reads one case file
(/input/<id>.in) - everything recorded up to the end of the release day's
first hour, that hour's count included - and writes one forecast file
(/output/<id>.out): for each of hours 2..24, five quantiles (5%, 25%, 50%,
75%, 95%) of that hour's concurrent viewers in the game's Twitch category.
Scored by mean pinball loss on the log scale (lower is better), so the
quantiles must be the ones the program really believes.

Method - unlock-aligned comparable log-ratio model
--------------------------------------------------
A launch day's shape is not a smooth diurnal curve: regions unlock at
rolling midnight, so viewers jump as each time zone wakes (New Zealand
first, then Europe, then the US East at 05:00Z and the US West at 08:00Z
for a November release). The shape also depends strongly on launch size:
an hour-1 count of a few hundred grows a hundredfold over the day, while
a launch that already opens at a million stays almost flat (across the
38 comparable launches in a case, the correlation between log hour-1
viewers and the log ratio at hour 19 is about -0.93).

So for each comparable launch c in the case file, the program takes its
release-day curve as log ratios L_c(h) = ln(v_c(h) / v_c(1)) and predicts
the target's log ratio at each hour h with a local-linear regression of
L_c(h) on ln(p1_c), kernel-weighted by closeness of ln(p1_c) to the
target's ln(p1). A second predictor adjusts the fit: the target's
pre-release level relative to its hour-1 print (log of the mean viewers
in the 48 hours before release day over the hour-1 print). Launches whose
audience was already there before release grow less; spike launches whose
hour 1 dwarfs their pre-release baseline grow more (correlation about
-0.4 with the size-adjusted residual).

The spread sigma(h) is the kernel-weighted residual spread of that fit
(so it widens over the day, as a real launch day's uncertainty does), and
the quantiles are q_tau(h) = p1 * exp(m(h) + z_tau * sigma(h)), with
z the standard normal's quantiles. The spread is scaled by ALPHA = 0.8,
tuned on honest leave-one-launch-out backtests (see README).

Nothing is hardcoded about any launch: every number the program uses
comes from the case file it is reading. It reads no network, uses only
the standard library, is deterministic (same bytes on every run), and
runs in well under a second.

Safety: any failure at all (bad input, no usable comparables, a
non-finite number anywhere) falls back to the house persistence
baseline - median = hour-1 print every hour, spread widening like a
random walk - so the program always writes a valid forecast.
"""

import json
import math
import os
import statistics
from datetime import date
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Local-linear kernel bandwidth, in ln(hour-1 viewers) units.
BANDWIDTH = 1.5
# Spread multiplier, tuned on leave-one-launch-out backtests.
ALPHA = 0.8
# Spread floor / cap, in log units.
SIGMA_MIN = 0.15
SIGMA_MAX = 1.6
# A comparable needs at least this many release-day hours (of 2..24) to count.
MIN_HOURS = 20
# A comparable needs at least this many pre-release points to carry a
# pre-release ratio; fewer and its ratio is treated as unknown.
MIN_PRE_POINTS = 4


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_DAY_CACHE = {}


def hour_index(stamp):
    """Hours since the epoch for an ISO UTC stamp like '2026-09-15T00:00:00Z'.

    Dates repeat across a case file's thousands of points, so the day part
    is parsed once and cached; the hour is read off the string.
    """
    day = stamp[:10]
    ordinal = _DAY_CACHE.get(day)
    if ordinal is None:
        y, m, d = int(day[0:4]), int(day[5:7]), int(day[8:10])
        ordinal = date(y, m, d).toordinal()
        _DAY_CACHE[day] = ordinal
    return ordinal * 24 + int(stamp[11:13])


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100)
    / 100 rounds it. A whole number comes back as an int, so json.dumps
    writes 1240 where JavaScript writes 1240 (not 1240.0)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def _weighted_mean(pairs):
    """Weighted mean of (value, weight) pairs; None when the weights are 0."""
    num = 0.0
    den = 0.0
    for value, weight in pairs:
        num += weight * value
        den += weight
    return num / den if den > 0 else None


# ---------------------------------------------------------------------------
# Reading the case
# ---------------------------------------------------------------------------

def comparable_curves(case):
    """One entry per comparable launch with a usable release day.

    Each entry is (name, p1, hours, pre_ratio): the launch's hour-1 print,
    its release-day hourly viewers {hour: viewers} for hours 1..24 (hour 1
    is day_start, 00:00 UTC of its release date), and its pre-release
    ratio - the log of its mean viewers over the 48 hours before release
    day, divided by its hour-1 print (None when too few pre-release
    points exist). A comparable whose release day is incomplete, or that
    has no hour-1 point (Grand Theft Auto VI's own pre-release history,
    for instance, ends before release day), is skipped.
    """
    curves = []
    for comp in case.get("comparables") or []:
        day_start = hour_index(comp["day_start"])
        by_hour = {}
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if viewers is None or viewers <= 0:
                continue
            offset = hour_index(point["from"]) - day_start
            if 0 <= offset < 24:
                by_hour[offset + 1] = viewers
        if 1 not in by_hour:
            continue
        if sum(1 for h in range(2, 25) if h in by_hour) < MIN_HOURS:
            continue
        p1 = by_hour[1]
        pre = [point["viewers"] for point in comp.get("hourly") or []
               if -48 <= hour_index(point["from"]) - day_start < 0
               and point.get("viewers") is not None]
        if len(pre) >= MIN_PRE_POINTS:
            pre_ratio = math.log((sum(pre) / len(pre)) / p1)
        else:
            pre_ratio = None
        curves.append((comp.get("name", "?"), p1, by_hour, pre_ratio))
    return curves


def target_pre_ratio(case, p1):
    """The target's own pre-release ratio, computed exactly like a
    comparable's: log of the mean viewers over the 48 hours before its
    release day (from history_hourly, falling back to history) divided by
    its hour-1 print. None when the case carries no pre-release points."""
    day_start = hour_index(case["day_start"])
    points = case.get("history_hourly") or case.get("history") or []
    pre = []
    for point in points:
        stamp = point.get("from") or point.get("at")
        viewers = point.get("viewers")
        if stamp is None or viewers is None:
            continue
        if -48 <= hour_index(stamp) - day_start < 0:
            pre.append(viewers)
    if len(pre) < MIN_PRE_POINTS:
        return None
    return math.log((sum(pre) / len(pre)) / max(1.0, p1))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _local_linear(xs, ys, ws, x0):
    """Weighted local-linear fit y ~ a + b*(x - x0); returns (a, b).

    Centring at the target's x0 makes a the fit's value at the target.
    Degenerate designs fall back to the weighted mean (b = 0).
    """
    mean_x = _weighted_mean(list(zip(xs, ws)))
    mean_y = _weighted_mean(list(zip(ys, ws)))
    if mean_x is None or mean_y is None:
        return None, None
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return mean_y, 0.0
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    return mean_y + slope * (x0 - mean_x), slope


def predict_log_ratios(curves, p1, target_pre):
    """Predict the target's release-day log ratios, hour by hour.

    Returns {hour: (median log ratio, sigma)} for hours 2..24. At each hour
    the median comes from a local-linear regression of the comparables'
    log ratios on ln(hour-1 print), kernel-weighted by size similarity
    (bandwidth BANDWIDTH in log units). A second step adjusts that fit by
    the pre-release ratio: the residual slope on (pre_ratio - target_pre)
    is estimated from the comparables that carry one (ridge 0.5 for
    stability) and applied at the target's own ratio, clamped to the
    comparables' range. The spread is the kernel-weighted residual spread
    of the final fit, floored and capped.
    """
    x0 = math.log(max(1.0, p1))
    out = {}
    for hour in range(2, 25):
        rows = []
        for _name, cp1, by_hour, pre in curves:
            if hour not in by_hour:
                continue
            x = math.log(max(1.0, cp1))
            y = math.log(by_hour[hour] / cp1)
            weight = math.exp(-0.5 * ((x - x0) / BANDWIDTH) ** 2)
            rows.append((x, y, weight, pre))
        if len(rows) < 5:
            # Too little to fit: fall back to the plain median and IQR.
            ys = sorted(y for _x, y, _w, _p in rows)
            if not ys:
                out[hour] = (0.0, 0.5)
                continue
            median = statistics.median(ys)
            if len(ys) >= 4:
                quartiles = statistics.quantiles(ys, n=4, method="inclusive")
                sigma = max(SIGMA_MIN, (quartiles[2] - quartiles[0]) / 1.349)
            else:
                sigma = 0.5
            out[hour] = (median, sigma)
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        ws = [r[2] for r in rows]
        centre, slope = _local_linear(xs, ys, ws, x0)
        preds = [centre + slope * (x - x0) for x in xs]
        if target_pre is not None:
            have_pre = [(r[3], y - p, w)
                        for r, y, p, w in zip(rows, ys, preds, ws)
                        if r[3] is not None]
            if len(have_pre) >= 5:
                zs = [z for z, _r, _w in have_pre]
                z0 = min(max(target_pre, min(zs)), max(zs))
                mean_z = _weighted_mean([(z, w) for z, _r, w in have_pre])
                mean_r = _weighted_mean([(r, w) for _z, r, w in have_pre])
                szz = sum(w * (z - mean_z) ** 2 for z, _r, w in have_pre) + 0.5
                szr = sum(w * (z - mean_z) * (r - mean_r)
                          for z, r, w in have_pre)
                coef = szr / szz
                centre += coef * (z0 - mean_z)
                preds = [p + (coef * (pre - mean_z) if pre is not None else 0.0)
                         for p, (_x, _y, _w, pre) in zip(preds, rows)]
        num = 0.0
        den = 0.0
        for w, y, p in zip(ws, ys, preds):
            num += w * (y - p) ** 2
            den += w
        sigma = math.sqrt(num / den) if den > 0 else 0.5
        out[hour] = (centre, min(max(sigma, SIGMA_MIN), SIGMA_MAX))
    return out


def forecast_case(case):
    """Forecast one case: {"hours": [{"hour": h, "q": [q05, ...]}, ...]}."""
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = comparable_curves(case)
    if not curves:
        raise ValueError("case has no usable comparables")
    log_ratios = predict_log_ratios(curves, p1, target_pre_ratio(case, p1))
    hours = []
    for hour in range(2, 25):
        median, sigma = log_ratios[hour]
        spread = sigma * ALPHA
        quantiles = [round2(p1 * math.exp(median + z * spread)) for z in Z]
        quantiles = [min(max(q, 1), 1_000_000_000) for q in quantiles]
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


# ---------------------------------------------------------------------------
# Fallback: the house persistence baseline
# ---------------------------------------------------------------------------

def persistence_forecast(case):
    """Median = hour-1 print every hour, spread widening like a random walk
    in log space: q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1)). This is
    the starter's forecast, used only when the model cannot run at all."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    p1 = max(1, first["viewers"]) if first is not None else 1
    hours = []
    for hour in range(2, 25):
        quantiles = [round2(p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)))
                     for z in Z]
        quantiles = [min(max(q, 1), 1_000_000_000) for q in quantiles]
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
            # Guard the contract before writing: 23 rows, five finite
            # non-decreasing quantiles each, within (0, 1e9].
            rows = forecast["hours"]
            assert len(rows) == 23
            for row in rows:
                qs = row["q"]
                assert len(qs) == 5
                assert all(isinstance(q, (int, float))
                           and math.isfinite(q) and 0 < q <= 1_000_000_000
                           for q in qs)
                assert all(qs[i] <= qs[i + 1] for i in range(4))
        except Exception:
            # Never crash, never write an invalid case: fall back to the
            # persistence baseline, which always satisfies the contract.
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = persistence_forecast(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError("forecast for %s exceeds 65,536 bytes" % path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
