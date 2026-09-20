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
