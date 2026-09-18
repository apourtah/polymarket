# Polymarket weather markets — "priced ≥96¢ and still lost"

Pulls every resolved Polymarket weather market with an end date on/after
2026-01-01, downloads the price history of each market's losing outcome, and
flags markets where the losing outcome was priced at 96¢ or higher.

## Pipeline

```
python3 fetch_markets.py                              # Gamma API -> data/markets.json
python3 fetch_prices.py                               # CLOB 5-min history, all markets -> data/prices/
python3 analyze.py                                    # stage 1: candidates -> out/candidate_ids.json
python3 fetch_prices.py --fine out/candidate_ids.json # CLOB 1-min history, candidates -> data/prices_1m/
python3 analyze.py                                    # stages 2-3: 1-min stats + trade tape -> out/
```

All fetchers are resumable (per-page / per-market caching).

## Data sources

* **Gamma** `GET /events/keyset?tag_id=84&closed=true&end_date_min=2026-01-01`
  (tag 84 = `weather`; keyset pagination via `after_cursor` — the plain
  `/events` offset endpoint caps at offset 2100).
* **CLOB** `GET /prices-history?market=<token>&startTs&endTs&fidelity=<min>`
  — midpoint price samples. Windows longer than ~14 days are rejected at fine
  fidelity, so long-lived markets are fetched in 14-day chunks.
* **Data API** `GET /trades?market=<conditionId>` — the actual fill tape, used to
  measure dollars traded at ≥96¢ (candidates only).

Only the **losing** token of each binary market is fetched: it is the only side
that can satisfy "≥96¢ and lost", and the two tokens are complements.

## Robustness filters (why not just `max_price >= 0.96`)

A single print at 96¢ in a thin book is not the same as the market sitting
there. `analyze.py` therefore records, per candidate:

| column             | meaning                                                          |
|--------------------|------------------------------------------------------------------|
| `max_price`        | highest midpoint of the losing outcome                           |
| `longest_run_ge96` | longest contiguous stretch (minutes) with midpoint ≥ 0.96        |
| `minutes_ge96`     | total minutes at ≥ 0.96                                          |
| `vol_ge96_usd`     | $ notional actually filled in the losing outcome at ≥ 0.96       |
| `n_trades_ge96`    | number of such fills                                             |
| `first_ge96` / `last_ge96` | when the losing side first / last sat at ≥ 0.96          |
| `hrs_before_end`   | hours from `last_ge96` to the market's end date                  |
| `hrs_before_close` | hours from `last_ge96` to `closedTime` (resolution)              |

and applies:

* **`flag`** — `longest_run_ge96 >= 30` min **or** `vol_ge96_usd >= $500`.
* **`flag_strong`** — `flag` and `hrs_before_end <= 6`: the losing side was
  still ≥96¢ close to resolution, not an early mis-price that corrected.

Thresholds are constants at the top of `analyze.py`.

## Outputs (`out/`)

* `all_markets_stats_5m.csv` — one row per market, coarse stats.
* `candidates_ge96.csv` — every market whose losing side ever hit ≥96¢, with all
  columns above plus `flag` / `flag_strong` / `pass_run` / `pass_vol` / `late`.
* `flagged.csv` — the `flag == True` subset.

## Caveats

* `prices-history` returns **midpoints**, not trades. The trade tape is the
  volume check; the midpoint series is the persistence check.
* Multi-bucket temperature events (e.g. nine 2°F buckets) put every bucket's
  **No** near 96–99¢ by construction, so "No ≥96¢ and lost" is common whenever
  an underdog bucket wins. Split on `losing_outcome` (Yes vs No) when reading
  the results; Yes-side losses are the rarer, more interesting kind.
* "Resolved since January" is implemented as event `endDate >= 2026-01-01`.
  Markets that ended in late December but resolved in January are not included.
* Markets not resolved to a clean 0/1 (e.g. 50/50 voids, or still under UMA
  proposal) are skipped.
