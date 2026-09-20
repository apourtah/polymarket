# Who has alpha in the temperature markets? (branch `wallet-alpha`, data Aug 7 – Sep 18 2026)

Taker-side tapes for every highest-temperature market of the 7 production cities: 3,311 markets, 517,823 fills, 9,594 wallets
(`fetch_wallet_tapes.py` → `data/tapes_w/`). Every fill is converted to a YES-equivalent position and attributed the P&L it
would earn held to resolution (`q · (won − p_yes)`), so a wallet's "P&L" here is the information content of its fills, not its
cash P&L if it scalps in and out. Scripts: `wallet_alpha.py` (scoring, per-day leaderboards), `wallet_timing.py` (feed-latency
and jump-lead forensics), `wallet_cluster.py` (same-person detection). Outputs in `out/wallets_*.csv`, `out/wallet_pairs.csv`,
`out/wallet_clusters.csv`, `out/wallet_fills.parquet`.

## 1. Alpha is real, persistent and concentrated

* 1,845 wallets traded ≥10 markets on ≥8 days; 982 are net positive; **26 have P&L > $1k with bootstrap p ≤ 0.05** over days.
* Split-half test (odd vs even days): of the top-20 wallets on odd days, **17 are positive on even days**; rank correlation
  across all 2,377 wallets is only 0.11 — alpha lives in a small head, the rest is noise.
* The top-20 wallets made **+$86k**; all 9,594 wallets together made +$71k on $8.9M volume, i.e. the rest of the market is
  the liquidity that pays them.
* Per-day top-10 persistence (43 days): sailor82 22 days, neo7777 21, **0xdd22 17**, fildoro 16, 0xf7a4 15, sleeper-service 15.

## 2. Three kinds of edge (and none of them is insider information)

**A. Faster observation feed (print snipers).** `HighTempTation` (0x6011), `LowTempTation` (0x8e40) and `ducky77` (0xd115)
place 66–77 % of their market-day fills **within 2 minutes of the METAR print that set a new daily max** (baseline 2.3 %),
median 69–73 s after the observation time — before the public APIs show it (3–7 min lag). HighTempTation buys NO on the
just-cleared bucket at ~96¢ (our retired `metar_no_bot` idea, executed fast enough): 312 fills, **100 % winning days**,
+$2.7k on $89k (+3 %). ducky77 buys YES on the confirmed bucket at ~76¢ and the price moves +14¢ in the next hour
(markout +0.142, the highest in the data). 0xe7c65c does the mirror (buys NO at YES≈30¢, price falls 11¢ within the hour).
This is infrastructure, not insider knowledge: a direct NOAA/ADDS feed polled every few seconds. Their ROI is small because the
trade is nearly riskless and everyone with the feed competes for it.

**B. Morning forecast models (our peer group).** Zero print-chasing, 40–90 % of dollars placed 00:00–09:00 local on the market
day or the evening before, 20–45 % ROI on $10–40k over 6 weeks, positive on 60–72 % of days:

| wallet | P&L | staked | ROI | days | day-win | p_boot | style |
|---|---|---|---|---|---|---|---|
| 0x56b381… | +$6.9k | $39.7k | +17 % | 25 | 72 % | 0.04 | 85 % long, avg 47¢, 28 % day-before |
| **0xdd22…** | +$5.7k | $29.1k | +20 % | 39 | 62 % | 0.10 | the account we reverse-engineered |
| BeefSlayer 0x331b | +$5.3k | $11.7k | **+45 %** | 40 | 50 % | 0.03 | 71 % long, avg 37¢, markout +0.06 |
| 0x9506… | +$5.3k | $24.6k | +21 % | 40 | 70 % | 0.01 | all day-of, avg 36¢ |
| fildoro 0x180e | +$4.8k | $26.7k | +18 % | 41 | 63 % | 0.02 | 89 % day-before, 91 % long |
| Weather-Guru 0xb6fb | +$2.2k | $6.9k | +32 % | 26 | 58 % | 0.00 | 83 % day-before |
| wxwizzard 0xb6af | +$2.5k | $33k | +8 % | 39 | 80 % | 0.00 | high volume, small edge |

Our evening bot's backtest on the same six weeks (+$367 – $754 on $1.7k at $10 clips, +21–44 %) is in this group's range;
the group's dollars-per-day are 10–50× ours.

**C. Volume grinders.** neo7777 (+$9.0k on $221k, +4 %), sailor82 (+$7.9k on $295k, +2.7 %, 5,753 fills), 0x122cb9
(+$4.0k on $67k, 6,594 fills of ~9 shares, 85 % winning days), Rexc8 (buys 80¢ favourites, 100 % winning days, +1.3 %).
Consistent, tiny margins, huge turnover — market-making-like behaviour on the taker side (sweeping stale quotes).

**Jump-leading** (fills in the 5 min before a ≥10¢ move): 0xe7c65c 44 % of fills, bellotas500 39 %, JapeththeGoat 31 %,
wxwizzard 24 % — but their direction is right only 47–60 % of the time, so they trade *at* volatile moments (prints), they
don't foresee them. No wallet in the data leads price jumps with a direction hit-rate that would suggest non-public information
(e.g. knowing the resolution source's reading before it is published).

## 3. Same-person wallet groups

Signals that work, in order of strength:
1. **Creation timestamps.** Polymarket's default display name is `0x<address>-<epoch ms>`; 1,522 wallets carry it. Wallets created
   minutes apart with identical fingerprints are one operator: a **12-wallet farm** created 02:01–02:33 UTC on May 15 2026
   (0x005e…, 0x16c7…, 0x1abc…, 0x1f85…, 0x3653…, 0x4e72…, 0x69f0…, 0x8b8b…, 0x9b12…, 0xb2e4…, 0xd717…, 0xeb68…): ~4,750 fills
   each, hour/city cosine 1.00, 560–790 co-fills per pair, P&L ±$500 each ($2.2k total on $422k) — a volume/points farm.
2. **Co-fills far above chance** (same market within 30 s, observed/expected under independence): `sailor82` fills alongside
   thunderfrog, berlinskies, av223, superhappymonkey, firstbroad, lunarshadow1, mryeth at **2,500–5,700× chance**; sailor82
   +$7.9k, the satellites −$110…+$530. Same pattern: fildoro ↔ yueyueyn/wfffkkl (2,700–3,800×), HighTempTation ↔ Happening9014
   (4,900×), Amano-Hina ↔ formon (14,259 co-fills). One operator, one signal, several wallets — or a taker splitting size
   across accounts.
3. **Fingerprint twins**: identical hour histogram, city mix and top-5 clip sizes (`fp ≥ 0.95`): quietparcel / northdrawer /
   plainfolder / graynotebook19 / mildwindow27 / god83743894 (1.8–2.2k fills each, ≈$0 P&L: another farm);
   e46m3 / JustClickeverywhere / gmgmgmgmg; Lucerys / Mysaria.
4. **Names**: HighTempTation / LowTempTation; `weather-highest-snipper-v2-api` / `Ah-weather-highest-snipper-v2-api`;
   SonofWeather / SoW-yiyi; andytest1/2/3.

Strict connected components (creation ≤1 h apart, or co-fills ≥30 at ≥8× chance with fingerprint ≥0.85, or fingerprint ≥0.95):
**39 clusters, 276 wallets**. Of the top-25 alpha wallets, 19 are singletons; the clustered ones are sailor82 (+xTriple7),
Amano-Hina (+formon), 0x122cb9 (+HotWeatherGirl), anonymous5474495 (in the 18-wallet farm), IngressDefender (+1).
The 0xdd22 account is a singleton here — no sibling wallet in these markets.

## 4. What else would be informative (not done)

* **On-chain funding graph** (Polygon): the first USDC deposit into each proxy and the EOA that owns it — wallets funded from
  the same address or in the same transaction batch are the same person; needs a Polygonscan/Alchemy key.
* **Maker side of the book.** The data-api tape is taker-only; the CLOB reveals maker identity only when a maker's order is
  hit. Combining both sides via the on-chain `OrderFilled` events (maker + taker addresses per fill) would show who *quotes*
  the 96–99¢ levels the snipers hit, and who is on the other side of the morning-model group.
* **Cross-category activity** (data-api `/activity?user=`): whether a wallet trades only weather (a specialist bot) or
  everything (a generalist) — specialists' fills carry more information.
* **Latency fingerprint**: seconds after the top of the hour for print snipers (feed identity: NOAA tgftp vs ADDS vs Synoptic),
  and modal inter-fill gap (timers: 0xdd22 ran a 30-min loop in May–Jul; 0183648392 runs a 28-min loop).
* **Resolution-source proximity**: the only true "insider" in these markets would be someone with the station's ASOS 1-minute
  data or the Weather Underground page cache ahead of the METAR — testable as fills that precede the *observation time*
  of a new max, not just its publication. None of the 40 wallets examined trades before the observation time.

## 5. What this means for our bot

* Our edge type (B) is shared by ~6 wallets moving $10–40k a week; the market absorbs them, so the edge is not about to
  vanish, but the 21:00–23:00 window is where they all buy, which is why 30–55¢ picks fill worse than mids suggest.
* The (A) snipers are why the print-driven NO idea failed for us and why HighTempTation earns 3 %: it is a latency race.
* If we ever want copy-signals, the wallets to watch per day are 0x9506, BeefSlayer, fildoro, 0x56b381 and 0xdd22 — their
  fills between 00:00 and 09:00 local, not the crowd's afternoon flow.

---

# Part 2 — Finding rotated wallets, and is following them worth it? (`wallet_reid.py`)

**Do they rotate?** In these six weeks, mostly no: 25 of the top-30 alpha wallets are active on 30–43 of the 43 days. Two
disappearances (0x56b381 +$6.9k, last fill Sep 4; formon, Sep 3) and one appearance (Weather-Guru, Aug 24). The hand-off search
(fingerprint of the stopped wallet vs every wallet that started afterwards) finds no convincing successor for 0x56b381 (best
similarity 0.92, to wallets with ≈$0 P&L); the one strong match is anonymous5474495 → 0x92b7b4c0 (0.988, started the day after
it stopped — and lost $1,054).

**Can one day of behaviour identify a wallet?** Daily fingerprint = local-hour bands, day-before share, long share, YES-price
bins, city mix, clip-size signature, activity level. Leave-one-day-out, identity hidden, ranked among every wallet active that
day (~600 candidates):

| footprint | top-1 | top-3 | top-10 | median rank |
|---|---|---|---|---|
| full day | **58 %** | 74 % | 87 % | 1 |
| first 5 fills | 16 % | 28 % | 43 % | 17 |

Per wallet (full day, top-1): sailor82 95 %, 0x122cb9 95 %, sleeper-service 93 %, anonymous5474495 88 %, neo7777 81 %,
fildoro 80 %, 0x56b381 68 %, 0x9506 60 %, Weather-Guru 58 %; but 0xdd22 35 %, BeefSlayer 27 %, HighTempTation 17 %
(its footprint is generic "print sniper"). Open-set detection ("this unknown wallet-day is alpha wallet X" if similarity ≥ τ):
τ = 0.97 → recall 37 % at 69 % precision; τ = 0.95 → 58 % / 39 %; τ = 0.93 → 70 % / 22 %.
So: a rotated wallet of a distinctive trader **can** be re-found by the end of its first day with ~60–95 % accuracy, but not
from its first few fills — too late to act on that day, useful for building the next day's watch-list.

**Is following them profitable?** Train Aug 7–27 (select 15 alpha wallets by P&L and t-stat), test Aug 28–Sep 18, $10 per
wallet-market, copied at the next print ≥ 60 s after their fill, fee included:

| strategy | clips | ROI | P&L | negative days / 22 |
|---|---|---|---|---|
| S1 static alpha ids | 3,315 | −7 % | −$2,293 | 17 |
| S2 yesterday's top-10 ids | 2,223 | −1 % | −$316 | 16 |
| S3 fingerprint-matched wallets (id-agnostic), τ 0.93 / 0.95 / 0.97 | 871 / 216 / 22 | −18 % / −29 % / −41 % | | |
| S4 fill-level "informed flow" model (no identity), edge ≥ 0.10 | 5,864 | 0 % | +$26 | 11 |
| copy every fill | 39,891 | −7 % | −$27k | 20 |

Per wallet, first fill per market, at their own price vs delayed (Aug 28–Sep 18): 0x9506 +12 % own / **+13 % at +1 h**;
Weather-Guru +20 % / +19 %; 0xdd22 +8 % / +14 %; fildoro +21 % / +1 % at 60 s (timing edge, evaporates);
ducky77 +16 % / −8 % (latency edge); HighTempTation +4 % / +1 %; BeefSlayer −9 % / −27 % (its +45 % was the training weeks);
neo7777 −21 % / −28 %; sailor82 +3 % / −3 %.

**Conclusions.**
1. Wallet rotation is rare here and, when it happens, re-identification from behaviour works for distinctive traders
   (60–95 % top-1 after one day) — good enough to maintain a watch-list across rotations, not to copy intraday.
2. Following alpha wallets does not pay: the group's edge is *price and time*, not the pick — by the next print the price
   has moved, and the selection itself is noisy (15 train-period winners: 8 lost out of sample). The only copyable wallets
   are the slow forecast-model traders (0x9506, Weather-Guru, 0xdd22, +13–19 % delayed), whose picks are the ones our own
   model already makes.
3. The id-free "informed flow" detector is exactly break-even: the tape does not carry a free signal beyond price.

## Part 3 — copy only what does not move the market (`wallet_copy_lowimpact.py`)

Impact per fill = move of the next print (≥60 s later) in the direction of the trade. In these thin books 50–82 % of every
wallet's fills are followed by a print ≥1¢ away; only 2 of 25 train-period alpha wallets have ≤50 % (IngressDefender,
PintouOClima). Test Aug 28–Sep 18, $10 clips at the next print:

| selection | clips | ROI |
|---|---|---|
| all alpha ids | 4,711 | −8 % |
| low-impact alpha wallets only | 307 | −5 % |
| alpha ids, copy only fills whose next print is unchanged (0¢) | 1,649 | **−15 %** |
| … within 1¢ / 2¢ | 3,497 / 4,004 | −18 % / −13 % |
| low-impact wallets AND unchanged print | 135 | −8 % |
| any wallet, unchanged print (baseline) | 26,364 | −14 % |

Filtering on "the price did not move after their fill" selects the *uninformative* fills — when the market did not react, the
market disagreed and was usually right. The impact IS the information; you cannot keep one without the other. Copying is
closed as a line of research.

## Part 4 — select on (alpha × low impact), copy every fill (`wallet_copy_all.py`)

74 train-period alpha candidates (≥8 markets, ≥6 days, P&L > $200, t > 0.8), ranked by four impact measures (share of fills
followed by a ≥1¢ move, mean/median directional move, $-weighted move). Every selection, copying **all** fills at the +60 s
print in the test period, is negative: −1 % to −4 % for the broad cuts, −7 % to −17 % as the alpha filter tightens
(t ≥ 2.5 & moved ≤ 0.5: 8 wallets, −17 %). Mirroring their dollar size (cap $50/fill): −9 %. The lowest-impact "alpha" wallets
are the market-making and farm bots (donthackme, highstakebet, Mysaria, the May-15 farm): 2–5k fills each in three weeks,
0 % to −1 % copied — their train-period P&L was noise on volume. Nothing in the (alpha, impact) plane is copyable.
