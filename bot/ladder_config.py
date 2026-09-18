"""Evening-dip ladder bot configuration. Secrets from the environment only."""
import os
DRY_RUN = os.environ.get("BOT_DRY_RUN", "1") != "0"

# --- strategy (from the dip backtest: YES touched >=99.5c, dips 18:00-19:59 local, bids 90-99c) ---
TOUCH = 0.995                    # YES midpoint must have reached this today before we ladder
WINDOW = (18, 20)                # rest bids only while local hour in [18, 20)
LEVELS = [round(0.90 + 0.01 * i, 2) for i in range(10)]   # 0.90 .. 0.99
STAKE_PER_LEVEL = 5.0            # $ per level  (10 levels -> $50 max per market)
MIN_ORDER_SHARES = 5             # Polymarket minimum
MAX_MARKETS_OPEN = 40            # cap on simultaneous ladders
MAX_DAILY_USD = 1000.0           # kill switch on filled notional per day
POLL_SECONDS = 20
DISCOVERY_SECONDS = 600

GAMMA = "https://gamma-api.polymarket.com"; CLOB = "https://clob.polymarket.com"; DATA_API = "https://data-api.polymarket.com"
CHAIN_ID = 137
PRIVATE_KEY = os.environ.get("POLY_PRIVATE_KEY"); FUNDER = os.environ.get("POLY_FUNDER"); SIGNATURE_TYPE = int(os.environ.get("POLY_SIG_TYPE", "0"))

STATE_FILE = "bot/ladder_state.json"; TRADE_LOG = "bot/ladder_trades.csv"; KILL_FILE = "bot/STOP"; LOG_FILE = "bot/ladder.log"
