"""Evening HRRR bot configuration. Secrets from the environment only."""
import json
import os
DRY_RUN = os.environ.get("BOT_DRY_RUN", "1") != "0"

# per-city legs (8-month backtest, backtest_evening.py MODES=...): "agree" = buy the agreed bucket on agreement nights;
# "ridge"/"ewma" = on disagreement nights buy that model's best bucket; "edge" = old avg-P edge rule (unused)
#   2026-09-26: the agree leg is dropped in Los Angeles, Austin and Seattle. Over Jan-Sep it returned 0.047 /
#   0.086 / 0.163 there against 0.49-1.06 for the disagree legs elsewhere, and at the ask (which is what we pay,
#   unlike the backtest's mid) LA/agree goes negative. Paired bootstrap P=99.9%, stable at every expanding-window
#   cutoff, wins all three folds (leg_audit.py). Seattle keeps an empty leg set rather than being removed, so its
#   forecast is still recorded and scored into the history -- it simply never trades.
MODES = {"Los Angeles": {"ewma"}, "Austin": {"ewma"}, "Chicago": {"agree", "ridge"},
         "Houston": {"ridge"}, "Dallas": {"ridge"}, "Seattle": set(), "Miami": {"agree"}}
CITIES = list(MODES)              # NYC/Denver/SF/Atlanta excluded (no edge in either leg)
MODEL_MAX_PRICE = 0.45            # ridge/ewma disagreement buys: YES ask <= this. 2026-09-19: 0.60 -> 0.45 (45-60c picks net $0 on $2.4k; 45-50c -13%)
MODEL_MIN_PRICE = 0.05
WINDOW = (21, 25)                 # act from 21:00 local until 01:00 the next day (25 = 01:00 next day); the start jitter keeps normal entries at 21:00-21:12, the tail lets a late restart still trade (edge study: still positive until ~02:00)
# --- rule (from the walk-forward lab) ---
AGREE_MIN_PRICE = 0.10            # agreement day: buy the agreed bucket only if this <= YES ask <= AGREE_MAX_PRICE
AGREE_MAX_PRICE = 0.53            # 2026-09-19: 0.35-0.60 -> 0.10-0.50 -> 0.10-0.53; 50-53c agree picks: 26 trades, 77% win, +43%; 53-60c negative (legs_backtest.py)
DISAGREE_EDGE = 0.10              # "edge" mode only: buy every bucket where avg(P_ewma, P_ridge) - ask >= this
DISAGREE_MAX_PRICE = 0.075
# --- NO leg (rule "H", no_compare.py, 2026-09-19): only on nights where a disagree-model YES leg (dis_ridge / dis_ewma) fired.
#   A: NO on the market favorite if it is WARMER than our YES bucket;  B: in ridge-mode cities, NO on the bucket 1 warmer
#   than the ridge pick.  Either way only if that bucket's YES ask is in [NO_LEG_MIN_YES, NO_LEG_MAX_YES] (NO entry ~0.46-0.66).
#   Backtest Jan18-Sep15 (35-55c): 76 nights, 84% win, +40%, 0 negative months. NO-fav paired with an agree-leg YES is ~0 -> not traded.
NO_LEG = True
NO_LEG_MIN_YES = 0.35; NO_LEG_MAX_YES = 0.55   # 0.30-0.35 was a coin-flip on fees (NO at ~69c, 72% win); 0.35-0.55: 76 nights, 84% win, +40%
NO_LEG_SHARE_MATCH = True         # size the NO leg to the YES leg's share count (the 0xdd22 sizing); else STAKE
STAKE = 12.0                      # base $ per bucket, before the edge tilt below.
# 2026-09-26: sized for a $200 wallet from a 134-point sweep over (EDGE_MULT, base stake, cap), scored by the
# P&L actually ACHIEVABLE on that wallet -- a cash-flow bootstrap over reshuffled orderings of the same trading
# days (3-day hold, $2 reserve) where a trade that cannot be funded simply does not happen. Running out of cash
# costs a missed opportunity and not a loss, and the wallet caps the downside at $200 whatever the stake, so
# the frontier climbs a long way: resampled P&L is $4233 at a $8.50 base, $4991 at $10, $5480 at $11, $5945
# here. It is the 10th percentile that sets the limit, and it falls off a cliff just above: p10 is $3489 at a
# $12 base and -$196 at $14, where one path in ten loses the whole wallet. Sized to sit below that cliff.
# 2026-09-26: size by the model's own edge instead of betting a flat stake on everything --
#   stake = STAKE * (1 + EDGE_MULT * (P_model - ask)), so a bucket the model likes far more than the price gets
#   up to MAX_PER_MARKET_USD and one it barely likes gets a couple of dollars. Largest single improvement found
#   (search.py b16/b17) and not a tuned constant: the response is monotone over every multiplier 1..8 and every
#   cap $15..$30, all of which beat the flat stake on both P&L and ROI across all three folds.
STAKE_MODE = "edge"               # "edge" | "flat"
EDGE_MULT = 6.0                   # 2026-09-26: 6 -> 4, mid-plateau. The multiplier sweep is monotone and flat
                                  # over 4..6 ($3979 / $4018 / $4024 standalone), so 4 is the same effect with a
                                  # gentler tilt: max stake is reached at edge +0.375 rather than +0.25.
MAX_PER_MARKET_USD = 36.0         # 3x the base stake
MAX_PER_CITY_DAY_USD = 42.0       # YES leg + share-matched NO leg
MAX_DAILY_USD = 190.0             # peak modelled day is $134, so this does not bind
MIN_ORDER_SHARES = 5
AUTO_REDEEM = True                # if a wallet's free USDC can't fund an order, redeem its resolved winners first (bot/redeem.py; gas = POL from the EOA)
CASH_RESERVE = 2.0                # keep this much USDC free after funding an order
# --- model ---
EWMA_GAINS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
RIDGE_ALPHA = 20.0
RIDGE_WINDOW = 180                # fit the ridge on the last N days only (the EWMA still sees all history: more is
                                  # strictly better there, model_variants_backtest.py EWMA_ONLY=1). The ridge has no
                                  # forgetting of its own, so it stayed anchored to a cold-biased spring: its bias ran
                                  # -0.33 (Feb-Jun) to +0.54 (Sep), by city Austin +1.59F, LA +1.25F, Miami +0.85F.
                                  # Out of sample (choose on Jan-Jun, score Jul-Sep, OOS_ONLY=1) every window in
                                  # 150-210d beat the unwindowed fit: +$142/+$226/+$204/+$216/+$164. 180 = mid-plateau,
                                  # not the argmax. NB in that same test, picking any setting by backtest P&L did NOT
                                  # transfer (rank corr +0.28; the in-sample winner lost $353) -- this lever is taken on
                                  # the measured drift plus the out-of-sample plateau, not on a sweep win.
MIN_HISTORY_DAYS = 60
FEATS = ['hrrr','dew','rhum','wind','cloud','rad','pres','precip','wdir_s','wdir_c','dpd','t_morn','cloud_morn','yday_max','yday_min','yday_err','e2','r5','r14','nbm_minus_hrrr','gfs_minus_hrrr','doy']
# --- endpoints ---
GAMMA = "https://gamma-api.polymarket.com"; CLOB = "https://clob.polymarket.com"; METAR = "https://aviationweather.gov/api/data/metar"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"; IEM_MOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
CHAIN_ID = 137
# --- wallet identity (environment only) ---------------------------------------------------
#   OWNER_PRIVATE_KEY   private key of YOUR OWN wallet (the EOA that owns your Polymarket account: the MetaMask account you
#                       connected, or the key exported from Settings for an email login). Not a Polymarket key — Polymarket
#                       never sees it; it only verifies signatures made with it.
#   POLYMARKET_PROXY    the proxy contract Polymarket created for you — holds your USDC and positions; the address shown on
#                       your profile page. Owned by the key above; has no key of its own.
#   POLYMARKET_LOGIN    how you log in: "email" (Magic proxy), "metamask" (browser wallet -> Gnosis Safe proxy), or "eoa"
#                       (no proxy: you trade straight from the owner wallet). Maps to py-clob-client signature types 1 / 2 / 0.
# Legacy names POLY_PRIVATE_KEY / POLY_FUNDER / POLY_SIG_TYPE are still read as a fallback.
LOGIN_TYPES = {"eoa": 0, "email": 1, "magic": 1, "metamask": 2, "browser": 2, "safe": 2}
def _login_to_sig(v):
    v = (v or "").strip().lower(); return LOGIN_TYPES.get(v, int(v) if v.isdigit() else 0)
PRIVATE_KEY = os.environ.get("OWNER_PRIVATE_KEY") or os.environ.get("POLY_PRIVATE_KEY")
FUNDER = os.environ.get("POLYMARKET_PROXY") or os.environ.get("POLY_FUNDER")
SIGNATURE_TYPE = _login_to_sig(os.environ.get("POLYMARKET_LOGIN") or os.environ.get("POLY_SIG_TYPE", "0"))
HISTORY = "data/evening_history.parquet"      # per city-day: hrrr, features, actual, err (seeded from the lab panel)
FORECASTS = "data/evening_forecasts.parquet"  # forecasts we made (to score once the day resolves)
STATE_FILE = "bot/evening_state.json"; TRADE_LOG = "bot/evening_trades.csv"; KILL_FILE = "bot/STOP"; LOG_FILE = "bot/evening.log"

# --- execution camouflage ---------------------------------------------------------------
LOOP_SECONDS = 60                 # order-management cadence inside the window
START_JITTER_MIN = (0, 12)        # each city-day starts at a random minute offset into the window (was 0-70; the backtest enters at ~21:35)
SKIP_PROB = 0.0                   # skip a city-day entirely with this probability
SIZE_JITTER = (0.8, 1.2)          # stake multiplier drawn per order
CHILD_ORDERS = (1, 1)             # split each stake into this many child orders
CHILD_GAP_MIN = (1, 3)            # minutes between children
REST_MIN = (2, 6)                 # rest a limit 1 tick under the ask for this long before crossing (was 5-20)
DEPTH_CAP = 1.00                  # market (cross) leg only: share of visible depth at <= ask + 1c we will take;
                                  # resting limit is full size regardless. 2026-09-26: 0.30 -> 1.00. The 0.30 was a
                                  # self-imposed footprint limit, not a market constraint, and it was the single
                                  # largest brake on execution: measured against 81 live books and 10089 exact
                                  # depth observations recovered from cleared sweep levels, raising it takes the
                                  # fill rate from 56% to 79% and realistic P&L from $3086 to $4176 -- more than
                                  # any stake increase. The cost is visibility: at this setting we take the whole
                                  # visible book at the touch and one cent behind it.
DECOY_PROB = 0.0                  # small non-strategy buy on the model's 2nd bucket, per city-day
DECOY_STAKE = (8, 20)
# several wallets: POLYMARKET_WALLETS = JSON list of {"owner_key":..., "proxy":..., "login":"email"|"metamask"|"eoa"}
# (legacy POLY_WALLETS with {"key","funder","sig_type"} still accepted). Falls back to the single wallet above.
def _norm(w): return {"key": w.get("owner_key") or w.get("key"), "funder": w.get("proxy") or w.get("funder"), "sig_type": _login_to_sig(str(w.get("login", w.get("sig_type", 0))))}
_wl = os.environ.get("POLYMARKET_WALLETS") or os.environ.get("POLY_WALLETS")
WALLETS = [_norm(w) for w in json.loads(_wl)] if _wl else ([{"key": PRIVATE_KEY, "funder": FUNDER, "sig_type": SIGNATURE_TYPE}] if PRIVATE_KEY else [{"key": None}])
# wallet rotation mode:
#   random      a wallet drawn at random per city-day (default; least structure in any single wallet's history)
#   per_city    fixed city -> wallet (WALLET_CITY_MAP, else hash of the city name). Each wallet's history is ONE city — most legible, only useful if the wallets can't be linked
#   per_signal  agree / disagree / decoy trades go to different wallets (WALLET_SIGNAL_MAP). Splits the profitable disagreement fills from the rest
#   round_robin cycle through wallets across city-days in order
#   per_day     one wallet for all cities on a given date (rotates daily)
WALLET_MODE = os.environ.get("WALLET_MODE", "random")
WALLET_CITY_MAP = json.loads(os.environ.get("WALLET_CITY_MAP", "{}"))          # e.g. {"Houston":0,"Austin":0,"Los Angeles":1,"Seattle":2}
WALLET_SIGNAL_MAP = json.loads(os.environ.get("WALLET_SIGNAL_MAP", '{"agree":0,"disagree":1,"decoy":2}'))
