"""Bot configuration. Secrets come from the environment, never from this file."""
import os

DRY_RUN = os.environ.get("BOT_DRY_RUN", "1") != "0"      # default ON: log intended orders, place nothing

# --- strategy ---------------------------------------------------------------
MAX_ASK = 0.95                  # only buy NO at or below this
MAX_POSITION_USD = 50.0         # per market, notional (price * shares)
MAX_DAILY_USD = 500.0           # global kill switch on total notional bought today
LOCAL_WINDOW = (11, 18)         # poll a city only while its local hour is in [11, 18)
POLL_SECONDS = 15               # METAR poll interval
BOOK_REFRESH_SECONDS = 60       # how often to re-check which buckets still have NO <= MAX_ASK
DISCOVERY_SECONDS = 600         # how often to re-pull the market list from Gamma
MAX_PRINT_AGE_MIN = 15          # ignore disqualifying prints older than this (startup backlog / stale data)
ASK_LEVELS = 2                  # size = sum of the first N ask levels (that are <= MAX_ASK)

# --- signal ------------------------------------------------------------------------
# Primary trigger: the Synoptic 5-minute stream (what NOAA's WRH page displays).
#   °F stations: a 5-min reading (whole °C -> °F) must exceed the bucket by >= MARGIN5_F
#                (backtest: +3 -> 96% win on the tradeable slice, +2 -> 80%).
#   °C stations: Synoptic carries only the half-hourly METARs themselves, so any
#                reading above the bucket is definitive (margin 1, no confirmation needed).
# Confirmation: an hourly METAR (T-group tenths, from aviationweather or a Synoptic
#   reading that carries tenths) above the bucket. Unconfirmed 5-min triggers may only
#   use TRANCHE_5MIN of the position; the rest is bought on confirmation.
MARGIN5_F = 3
MARGIN5_C = 1
TRANCHE_5MIN = 0.5              # fraction of MAX_POSITION_USD allowed before hourly confirmation
HOURLY_POLL_SECONDS = 15        # aviationweather poll (confirmation path; ~3.5 min behind the obs)
SYNOPTIC_TOKEN = os.environ.get("SYNOPTIC_TOKEN", "7c76618b66c74aee913bdbae4b448bdd")   # weather.gov's public WRH token
SYNOPTIC = "https://api.synopticdata.com/v2/stations/timeseries"
SYNOPTIC_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36",
                    "Referer": "https://www.weather.gov/wrh/timeseries?site=klga",
                    "Origin": "https://www.weather.gov", "Accept": "application/json"}
MIN_ORDER_SHARES = 5            # Polymarket minimum
MAX_FILL_LOOPS = 10             # safety cap on buy-recheck-buy iterations per print

# --- endpoints ---------------------------------------------------------------
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
METAR = "https://aviationweather.gov/api/data/metar"
CHAIN_ID = 137

# --- credentials (only needed when DRY_RUN=0) -----------------------------------
PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY")          # wallet private key (hex)
FUNDER = os.environ.get("POLY_FUNDER")                    # proxy/funder address if using a Polymarket proxy wallet
SIGNATURE_TYPE = int(os.environ.get("POLY_SIG_TYPE", "0"))  # 0 = EOA, 1 = Magic/email proxy, 2 = browser-wallet proxy

# --- files -----------------------------------------------------------------------
STATE_FILE = "bot/state.json"
TRADE_LOG = "bot/trades.csv"
KILL_FILE = "bot/STOP"          # touch this file to stop trading (polling continues, no orders)
