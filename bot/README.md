# METAR → NO bot

Buys NO on a "Highest temperature in <city>" bucket the moment an hourly METAR
print from the market's resolution station rules that bucket out.

## Run

```
pip install --user py-clob-client
BOT_DRY_RUN=1 python3 bot/metar_no_bot.py                      # default: logs what it would do
BOT_DRY_RUN=0 POLY_PRIVATE_KEY=0x... [POLY_FUNDER=0x... POLY_SIG_TYPE=1|2] python3 bot/metar_no_bot.py
touch bot/STOP                                                  # stop placing orders (polling continues)
```

`POLY_SIG_TYPE`: 0 = plain wallet (EOA) that holds the USDC; 1 = Polymarket
email/Magic proxy; 2 = browser-wallet proxy. For 1/2 set `POLY_FUNDER` to the
proxy address shown on your Polymarket profile. The wallet must have USDC.e on
Polygon and have approved the CTF exchange (done automatically the first time
you trade on polymarket.com).

## What it does, in order

| every | step |
|---|---|
| 10 min | **discover** — Gamma events (tag 84, open) titled "Highest temperature in …"; from the market text: station ICAO, unit (°F/°C), the observation day, each bucket's upper bound, the NO token id. "X or higher" buckets are excluded (heat can't rule them out). |
| 60 s | **candidates** — for cities whose local hour is in `[11, 18)` and whose market day is today: read each bucket's NO book; keep those with best ask ≤ `MAX_ASK` and position < `MAX_POSITION_USD`. Cities with none are not polled. |
| 15 s | **poll** — one `aviationweather.gov` METAR request for all active stations. Each unseen observation → rounded temp in the market's unit (T-group tenths, half-up — the rule verified against 606/608 resolved markets). If it's a new daily high, every bucket with `prev_max ≤ hi < temp` is newly disqualified → `execute()`. Prints older than `MAX_PRINT_AGE_MIN` (startup backlog, feed hiccups) are logged and skipped. |
| on trigger | **execute** — read the live NO book; take the first `ASK_LEVELS` (=2) levels with price ≤ `MAX_ASK`; `size = min(sum of their sizes, remaining room / price)`; place a **FAK** limit at the worst of those levels (fills what's there at or below, cancels the rest); re-read the book; repeat until no ask ≤ `MAX_ASK`, room is used up, or `MAX_FILL_LOOPS`. Then back to polling. |

State (positions per market, daily notional) persists in `bot/state.json`;
every fill goes to `bot/trades.csv`; everything is logged to `bot/bot.log`.

## Risk controls

* `MAX_POSITION_USD` per market, `MAX_DAILY_USD` global, `bot/STOP` kill file.
* Orders are FAK, never resting — nothing is left on the book.
* Minimum order 5 shares (Polymarket rule); anything smaller is skipped.
* `MAX_PRINT_AGE_MIN` prevents trading on stale prints. A bucket that was ruled
  out long ago but still shows a cheap NO is suspicious (corrected obs, station
  outage, resolution on the Weather Underground fallback) — the backtest's
  ~0.25% reversal rate lives there.

## Known limits / things to watch live

* **Latency**: the event study showed the market reprices within ~2–4 min of
  the observation *timestamp*. METAR publication itself lags the timestamp by
  1–3 min, so 15 s polling is the right order of magnitude but the win is in
  being early; a websocket book feed and a tighter poll (5 s) are the obvious
  upgrades.
* **Depth is untested**: the backtest had no book depth. Start with a small
  `MAX_POSITION_USD` and read `trades.csv` for a week before scaling.
* **Reversals**: 0.24% of rule-out prints did not resolve NO. Position sizing,
  not detection, is the defence.
* The aviationweather `temp` field already carries the T-group tenths; the
  raw-METAR parse is a fallback for records without it.
* Only "Highest temperature" markets. "Lowest temperature" markets would need
  the mirror rule (a print *below* the bucket's lower bound).

---

# Evening-dip ladder bot (`ladder_bot.py`)

The other strategy that survived backtesting (`dip_sweep.py`, `out/dip_sweep.parquet`):
a YES that has already touched 99.5¢ and gets sold down between 18:00 and 19:59 local
(after the day's high is in) almost always recovers — 0 reversals in 64–268 fills per bid
level, vs 10–13% reversals for the same dips at 14:00–17:00.

Rule: for each of today's buckets whose YES midpoint has touched `TOUCH` (99.5¢), while the
city's local hour is in `WINDOW` (18–20), rest GTC limit BUY YES at every `LEVEL`
(90¢…99¢), `STAKE_PER_LEVEL` ($5) each — $50 max per market. Cancel what's unfilled at
20:00 local; hold fills to resolution.

```
BOT_DRY_RUN=1 python3 bot/ladder_bot.py        # paper: fills inferred from the public trade tape
BOT_DRY_RUN=0 POLY_PRIVATE_KEY=0x... python3 bot/ladder_bot.py
```
State `bot/ladder_state.json`, fills `bot/ladder_trades.csv`, log `bot/ladder.log`, kill file `bot/STOP`.
Expect ~8 fills/month at the 90¢ level across all cities, more at higher levels; it's a
patience strategy — small, positive, and it doesn't need to be faster than anyone.

---

# Evening HRRR bot (`evening_bot.py`, `evening_config.py`)

The forecast-based strategy (see `correction_lab.py`, `out/correction_trades.parquet`). Nightly per city
between 21:00 and 23:00 local — the window in which the market has not yet absorbed the latest short-range run:

1. Latest complete extended HRRR run (00/06/12/18Z, AWS via herbie) → afternoon max at the station.
   Weather fields from Open-Meteo (`gfs_hrrr`), METAR max/min for today, NBM and GFS-MOS station maxes from IEM.
2. Two models refit on the history panel (`data/evening_history.parquet`, seeded Jun–Sep):
   EWMA of the station's HRRR error (gain tuned per station) and a pooled ridge regression of the error on the
   weather fields + lags + city.
3. P(bucket) for each of tomorrow's buckets under each model (normal, model residual sd).
4. **Agree** (same best bucket): buy it if the ask ≤ 60¢ (+23% in the lab).
   **Disagree**: buy every bucket where avg(P) − ask ≥ 10 pts (+90% in the lab; the disagreement *is* the signal).
5. FAK at the ask, $50 ($100 if edge ≥ 20 pts); caps $100/market, $200/city-day, $800/day; `bot/STOP` halts.
6. Each morning the previous day's forecast is scored against the METAR max and appended to history, so the
   models keep learning.

Cities: Austin, Houston, Los Angeles, Seattle (Miami/Atlanta showed no edge; SF's HRRR grid point is off by ~7°F).
Run: `BOT_DRY_RUN=1 python3 bot/evening_bot.py` (paper) — logs `bot/evening.log`, fills `bot/evening_trades.csv`.

### Execution camouflage (evening bot)
The fill history of a single wallet is public and is exactly how strategies get reverse-engineered
(we did it). `evening_config.py` therefore randomizes everything that isn't the edge itself:
`START_JITTER_MIN` (random start inside the window), `SKIP_PROB` (skip a city-day), `SIZE_JITTER` and
`CHILD_ORDERS`/`CHILD_GAP_MIN` (irregular sizes, split fills minutes apart), `REST_MIN` (rest a limit one
tick under the ask, then cross), `DEPTH_CAP` (never take more than 30% of visible depth), `DECOY_PROB`
(small buy on the 2nd bucket), and `WALLETS` (JSON list in `POLYMARKET_WALLETS`; a wallet is drawn per city-day).

### Wallet identity (evening bot) — what the three values are
- `OWNER_PRIVATE_KEY` — the private key of **your own wallet** (EOA): the MetaMask account you connected to
  Polymarket, or the key exported from Polymarket *Settings* for an email login. It is not a Polymarket key;
  Polymarket never holds it, it only verifies signatures made with it. It signs orders and pays gas for redemptions.
- `POLYMARKET_PROXY` — the proxy contract Polymarket deployed for you (the address on your profile page). It holds
  the USDC and the positions. It has no key; it executes what its owner (the key above) signs. The exchange can
  move funds out of it only against an order you signed, via the approvals set when you first traded.
- `POLYMARKET_LOGIN` — `email` (Magic proxy), `metamask` (browser wallet → Gnosis Safe proxy) or `eoa` (no proxy).
Use a dedicated wallet holding only the trading float. Redemptions: `python3 bot/positions.py --redeem [--live]`;
the bot also redeems automatically when a wallet is short of cash for an order (`AUTO_REDEEM`).
Wallets funded from one source are linkable on-chain; rotate them and fund them separately. Redeem
winnings by hand at irregular times rather than on a schedule.
