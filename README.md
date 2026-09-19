# Polymarket weather — evening HRRR bot

A bot that trades Polymarket's daily **"Highest temperature in \<city\>"** markets the evening before,
in the window (21:00–23:00 local) where the 00Z HRRR run is out but the market has not yet fully
absorbed it. Two bias-corrected forecasts vote on the winning 2°F bucket; the bot buys YES where the
backtest says that vote has an edge, holds to resolution, and disguises its footprint.

Everything else in this repo (data fetchers, labs, backtests, two retired bots) exists to build and
validate that bot.

## How the edge works

1. **Forecast** — the 00Z HRRR run's afternoon max (11–18 local) at the market's resolution station,
   pulled from AWS (`noaa-hrrr-bdp-pds`) via herbie.
2. **Two corrections** of that number, refit every night on all history to date:
   * **EWMA** — a per-station bias with a tuned gain (persistent local effects: sea breeze, siting).
   * **Ridge** — a pooled regression on HRRR fields, NBM / GFS-MOS station maxes, lags and city
     dummies (synoptic regimes: fronts, advection, cloud).
3. **Bucket probabilities** — each model's corrected max ± its recent residual SD → P(bucket).
4. **Per-city rule** (`MODES` in `bot/evening_config.py`), chosen on an 8-month walk-forward backtest:

   | city | models agree → buy the agreed bucket (10–53¢) | models disagree → buy … (5–60¢) |
   |---|---|---|
   | Los Angeles, Austin | yes | EWMA's bucket |
   | Chicago | yes | ridge's bucket |
   | Houston, Dallas | — | ridge's bucket |
   | Seattle, Miami | yes | — |

   **NO leg** (rule "H", from the 0xdd22 account's fill history; `legs_backtest.py`, `no_sweeps.py`, `no_compare.py`): only on
   a night where a disagree-model YES leg fired — (A) NO on the market favorite if it is warmer than our YES bucket, and
   (B) in ridge cities NO on the bucket one warmer than the ridge pick — when that bucket's YES price is 35–55¢ (NO bought at ~46–66¢), sized to the
   YES leg's share count. Backtest: 76 nights, 84% win, +40%, 0 negative months. Fading the favorite next to an *agree* YES,
   shorting the runner-up, or shorting buckets the models call overpriced all failed, as did every second-YES-bucket idea.

   Ridge cities are the ones where the day's max is set by the synoptic pattern; EWMA cities the ones
   with a stable station bias. NYC, Denver, SF, Atlanta showed no edge in either leg.
5. **Hold to resolution.** An exit sweep (take-profit 1.1×–10×, absolute levels, half exits,
   stop-losses) found nothing that beats holding: after the 21:00 entry the intraday price path is
   close to fair.

Backtest, Jan 18 – Sep 15 2026, $50 clips, mids +1¢, taker fees, no fills below 5¢: YES legs **+$12.0k on $36k (+34%)**,
NO leg +$1.5k on $5.4k (+29%), every month positive (`legs_backtest.py`, `no_compare.py`). The per-city split was
selected on the same data, so plan on ~+20–25%. Resolution rule (max over hourly METARs, T-group tenths
→ °F, half-up) verified against 606/608 resolved markets.

## Run

```bash
pip install --user py-clob-client herbie-data web3 scipy scikit-learn pandas requests

# paper (default) — logs every plan/decision/fill with [DRY]
BOT_DRY_RUN=1 nohup python3 bot/evening_bot.py >> bot/evening.out 2>&1 &

# live — three values, from the environment only (see "Wallet" below)
#   OWNER_PRIVATE_KEY   POLYMARKET_PROXY   POLYMARKET_LOGIN=metamask|email|eoa
BOT_DRY_RUN=0 nohup python3 bot/evening_bot.py >> bot/evening.out 2>&1 &

tail -f bot/evening.log                       # what it's doing
python3 bot/positions.py --watch 60 [--paper] # live positions, observed high so far, P&L
python3 bot/positions.py --redeem [--live]    # claim resolved winners
touch bot/STOP                                # kill switch (stops new orders)
```

## Wallet

* `OWNER_PRIVATE_KEY` — the private key of **your own** wallet: the EOA that owns your Polymarket
  account (the MetaMask account you connected, or the key exported from Polymarket *Settings* for an
  email login). It is not a Polymarket key; Polymarket only verifies signatures made with it.
* `POLYMARKET_PROXY` — the proxy contract Polymarket deployed for you (the address on your profile
  page). It holds the USDC and positions and has no key of its own; it executes what its owner signs.
* `POLYMARKET_LOGIN` — `metamask` (browser wallet → Gnosis Safe proxy), `email` (Magic proxy) or `eoa`.

Use a dedicated wallet holding only the trading float; the signing wallet needs ~1 POL for redemption
gas. No website login is needed after setup: API credentials are derived from the key at startup.
Secrets are read from the environment only (`.env` files are git-ignored).

## Sizing

Positions from two consecutive nights overlap (yesterday's resolve in the afternoon, tonight's go in at
21:00), so the float must cover ~13 clips open at once plus the drawdown (~12 clips):
**wallet ≈ 20 × `STAKE`**. Current config is `STAKE=10` for a $200 wallet (expect ~$150–300 / month);
`STAKE=60` / ~$1,200 wallet for ~$1k / month. Scale `STAKE`, `MAX_PER_MARKET_USD`,
`MAX_PER_CITY_DAY_USD`, `MAX_DAILY_USD` together. When a wallet can't fund an order the bot redeems its
resolved winners first (`AUTO_REDEEM`).

## Execution & camouflage

A wallet's fill history is public and is exactly how strategies get reverse-engineered (we did it to
someone else — see `trader_deep.py`). So nothing that isn't the edge is regular: random start minute
inside the window, jittered sizes, orders split into children minutes apart, a limit resting one tick
under the ask for a random 5–20 min before crossing (cross capped at 30% of visible depth), optional
decoy buys, and wallet rotation (`POLYMARKET_WALLETS`, `WALLET_MODE` = random / per_city / per_signal /
round_robin / per_day). Redeem at irregular times.

## Repo map

| | |
|---|---|
| `bot/evening_bot.py`, `bot/evening_config.py` | the bot and every knob (rule, sizing, camouflage, wallets) |
| `bot/redeem.py` | on-chain redemption for EOA / Magic-proxy / Safe wallets; balance check |
| `bot/positions.py` | positions CLI (`--watch`, `--paper`, `--wallet`, `--redeem`) |
| `backtest_evening.py` | walk-forward backtest of the YES rule (env: `START END CITIES POOL MODES MIN_PX SLIP` + rule overrides); dumps every priced bucket per night to `out/backtest_evening_buckets.parquet` |
| `legs_backtest.py`, `no_sweeps.py`, `no_neighbor_sweep.py`, `no_compare.py`, `no_leg_test.py` | leg tests on that dump: YES band, extra YES buckets, NO on favorite / neighbours / runner-up, rule comparison |
| `daybefore_backtest.py`, `fetch_hrrr_daybefore.py` | day-before forecasts (12Z/18Z NBM, GFS-MOS, HRRR) vs day-before prices — no edge found with MOS |
| `exit_sweep.py` | take-profit / stop-loss sweep on backtest trades |
| `correction_lab.py`, `low_lab.py`, `low_market.py` | model comparison lab; the (rejected) lowest-temperature variant |
| `fetch_hrrr*.py`, `fetch_metar.py`, `fetch_markets.py`, `fetch_prices.py` | data archives → `data/` (git-ignored, ~5 GB) |
| `trader_deep.py`, `favbias.py`, `coolbias.py`, `signal_study.py`, `event_study.py`, `dip_*.py`, `sim_pnl.py` | earlier studies: the 0xdd22 account, favorite bias, print-driven and dip strategies |
| `bot/metar_no_bot.py`, `bot/ladder_bot.py` | retired bots (print-driven NO, dip ladder) — see `bot/README.md` |
| `docs_ge96_study.md` | the original "priced ≥96¢ and still lost" study this repo started as |

## Data & history

`data/evening_history.parquet` is the model's memory: one row per station-day with the 00Z HRRR max,
features and the realized METAR max (11 US stations, Nov 2025 →). The bot appends to it each morning
(`score_pending`). Rebuild from scratch with `fetch_hrrr_back.py`, `fetch_metar.py`, the Open-Meteo
`previous-runs` and IEM MOS pulls (see `backtest_evening.py` for the panel construction).

## What was tried and rejected

* Print-driven NO buying on METAR (market reprices 1–2 min after the obs via faster feeds).
* Dip ladders at 90–99¢ after a 99.5¢ touch; morning forecast-vs-price strategies (market wins).
* Lowest-temperature markets (only NYC/Miami liquid, already priced at model skill).
* EU/Asia stations with regional models (no translation of the evening inefficiency).
* Any price-based exit.
