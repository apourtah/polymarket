# Does the evening rule work outside the US? (branch `global-cities`)

Motivation: the wallets that make $10–30k a month run a forecast-model taker strategy across 30–50 cities; ours runs in 7.

## Setup
* 37 non-US cities (all °C, 1-degree buckets, resolved from NOAA/METAR in whole °C), Jun 15 – Sep 18 2026: 37,884 markets,
  10-minute YES histories, 3,367 city-nights with a priced book at **21:35 local the evening before** (`global_fetch.py`).
* Actuals: METAR daily max (whole °C) from the existing 48-station archive.
* Forecasts (`global_forecasts.py`, `global_regional.py`): Open-Meteo previous-runs `previous_day1` afternoon max for
  ECMWF-IFS 0.25, GFS, ICON, plus the regional models where they exist — ICON-EU/D2, AROME-HD, ARPEGE, UKMO 2 km, HARMONIE
  (KNMI/DMI), MetNo, JMA MSM/GSM, CMA GRAPES, GEM regional/HRDPS, BoM. Raw ECMWF MAE 0.6 °C (Madrid) to 2.2 °C (Jeddah),
  warm biases of 1–2 °C in the tropics and Munich/Chongqing.
* Same machinery as the bot (`global_backtest.py`, `global_backtest2.py`): per-city EWMA bias correction, pooled ridge on the
  primary model + differences to every other model + ECMWF surface fields, nightly walk-forward refit, same legs
  (agree 10–53¢, model pick 5–45¢, NO leg H 35–55¢), $50 clips, mids +1¢, taker fee. v1 primary = ECMWF; v2 primary = the
  model with the lowest MAE in a pre-window calibration (May 25 – Jun 14).

## Result: no edge

| | EWMA bucket hit | ridge hit | **market favorite hit** | all legs everywhere | out-of-sample (legs picked Jun 15–Aug 1, tested Aug 2–Sep 18) |
|---|---|---|---|---|---|
| v1 ECMWF primary | 33 % | 40 % | **44 %** | −4 % (3,655 legs) | −7 % |
| v2 best regional primary | 34 % | 39 % | **44 %** | −1 % (3,639 legs) | −5 % |

By price band the YES legs lose everywhere above 20¢ (20–30¢ −4 %, 30–40¢ −1 %, 40–53¢ −10 %); the only positive band is
<10¢ (+28 %), which is the synthetic-mid tail we already distrust. Picks that are the favorite: −5 %; not the favorite: +2 %.
Per city a few look good in-sample (Warsaw +54 % agree, Wellington +150 % ridge, Cape Town +71 % ridge) and none of them
survive the split: the selected-legs test is −5 % with a −$2.5k month.

**Why the US works and this doesn't.** In the US the bot trades the **00Z HRRR run**, a 3-km model that is published at
~20:30 local the evening before and that the book has not absorbed by 21:00 — a freshness edge. Everything available for the
other cities at 21:35 local (12Z global runs, previous-day regional runs) is hours old and already in the price: the favorite
beats our best corrected model by 5–10 points in every region. The global winners (fildoro, 0x9506, TunSahur, opopv.) trade
at 00:00–09:00 local, i.e. **after the 00Z regional runs** (ICON-D2/AROME/UKMO/JMA-MSM at ~02:00–04:00 local), and much of
their P&L is intraday.

## What would be needed to go global
The 00Z regional runs are not archived anywhere we can reach (Open-Meteo's previous-runs API only gives whole-day lags; DWD/
Météo-France/JMA open data keep ~24 h). To test the "fresh 00Z regional run at 02:00–04:00 local" version honestly we would
have to **record those runs nightly from now** (a small fetcher per region) and paper-trade it for 6–8 weeks. That is a
data-collection project, not a backtest, and it would also need a night-time execution window per continent.
Until then the breadth lever is closed: the 7-city HRRR bot is the only version with a demonstrated edge.
