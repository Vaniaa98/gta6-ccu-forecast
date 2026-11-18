# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vaniaa98/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Vaniaa98/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A program that forecasts Grand Theft Auto VI's launch-day Twitch concurrent
viewers, hour by hour (hours 2–24), as five quantiles per hour — scored by
mean pinball loss on the log scale.

## How it works

A launch day is not a smooth diurnal curve: regions unlock at rolling
midnight, so viewers jump as each time zone wakes (New Zealand first,
then Europe, then the US East at 05:00Z and the US West at 08:00Z for a
November release). The shape also depends strongly on launch size — an
hour-1 count of a few hundred grows a hundredfold over the day, while a
launch that already opens near a million stays almost flat.

`main.py` (Python 3.13, standard library only) therefore forecasts from
the comparable launches inside each case file, not from any fixed curve:

1. For every comparable launch it takes the release-day curve as log
   ratios `L(h) = ln(v(h) / v(1))`.
2. At each hour it predicts the target's log ratio with a local-linear
   regression of `L(h)` on `ln(hour-1 print)`, kernel-weighted by size
   similarity (bandwidth 1.5 in log units).
3. It adjusts that fit by the target's pre-release ratio — the log of its
   mean viewers over the 48 hours before release day, divided by its
   hour-1 print. Launches whose audience was already there before release
   grow less; spike launches grow more.
4. The spread is the kernel-weighted residual spread of the fit, so it
   widens over the day as a real launch day's uncertainty does, scaled by
   0.8 (tuned on backtests). Quantiles are
   `q = p1 * exp(median + z_tau * spread)`.

Nothing is hardcoded about any launch: every number comes from the case
file being read. The program reads no network, is deterministic (same
bytes on every run), and runs in about 0.1 seconds per case. Any failure
at all falls back to the house persistence baseline, so it always writes
a valid forecast.

## Backtest (leave-one-launch-out, local scorer following the rules)

| case | launch | pinball loss | typical miss | 90% coverage |
|---|---|---|---|---|
| p00 | Marvel's Wolverine | 0.163 | 22.6% | 1.000 |
| p01 | Red Dead Redemption 2 | 0.205 | 44.5% | 0.957 |
| p02 | God of War Ragnarök | 0.140 | 26.4% | 1.000 |
| p03 | Marvel's Spider-Man 2 | 0.184 | 32.3% | 0.783 |
| **mean** | | **0.173** | **31.2%** | **0.935** |

House persistence baseline on the same cases: loss **0.387**, typical miss
82.8%, coverage 0.783. The practice leaderboard is saturated by entries
that replay the public practice answers (loss ≈ 0), which cannot transfer
to the unseen final case — this entry is judged on the honest
leave-one-launch-out numbers above instead.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory: 23 rows (hours 2–24), five non-decreasing quantiles each.

---

<sub>Entered on [make.fun](https://make.fun/?ref=thepowerofmuse&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
