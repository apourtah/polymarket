"""Evening-dip ladder: rest small YES bids under a bucket the market is already sure of.

Rule (from the dip backtest):
  - market: today's "Highest temperature in <city>" buckets
  - condition: YES midpoint has touched >= TOUCH (99.5c) at some point today
  - window: local hour in WINDOW (18:00-19:59) — after the day's high is in
  - action: rest GTC limit BUY YES at every LEVEL (90c..99c), STAKE_PER_LEVEL $ each
  - at window end (or market close): cancel whatever is unfilled; fills are held to resolution

Dry run (default) places nothing; it detects "would have filled" from the public trade tape:
a taker SELL of YES at or below a level (or taker BUY of NO at or above 1-level) after the
ladder went up counts as a fill of that level — the same rule the backtest used.

Run:  BOT_DRY_RUN=1 python3 bot/ladder_bot.py
      BOT_DRY_RUN=0 POLY_PRIVATE_KEY=0x... [POLY_FUNDER=... POLY_SIG_TYPE=1|2] python3 bot/ladder_bot.py
"""
import json, os, re, sys, csv, time, logging, datetime as dt
from zoneinfo import ZoneInfo
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import ladder_config as C
from sweep_local import TZ
TZ = {**TZ, "NYC": "America/New_York", "New York": "America/New_York"}

log = logging.getLogger("ladder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(C.LOG_FILE)])
for n in ("httpx", "httpcore", "urllib3"): logging.getLogger(n).setLevel(logging.WARNING)
TITLE_RE = re.compile(r"^Highest temperature in (.+?) on ")

def get_json(url, params=None, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200 and r.text.strip(): return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as ex: last = repr(ex)[:100]
        time.sleep(2)
    raise RuntimeError(f"{url.split('/')[2]}: {last}")

class Ladder:
    def __init__(self):
        self.markets = {}                 # cid -> dict(city, tz, day, question, yes_token, no_token)
        self.state = json.load(open(C.STATE_FILE)) if os.path.exists(C.STATE_FILE) else {"ladders": {}, "positions": {}, "daily": {}}
        self.touched = {}                 # cid -> bool (YES midpoint reached TOUCH today)
        self.t_disc = 0
        self.client = self._client()

    def _client(self):
        from py_clob_client.client import ClobClient
        if C.DRY_RUN or not C.PRIVATE_KEY:
            log.info("DRY RUN: no orders will be placed"); return ClobClient(C.CLOB)
        c = ClobClient(C.CLOB, key=C.PRIVATE_KEY, chain_id=C.CHAIN_ID, signature_type=C.SIGNATURE_TYPE, funder=C.FUNDER)
        c.set_api_creds(c.create_or_derive_api_creds()); log.info("LIVE: %s", c.get_address()); return c
    def save(self):
        json.dump(self.state, open(C.STATE_FILE + ".tmp", "w"), indent=1); os.replace(C.STATE_FILE + ".tmp", C.STATE_FILE)
    def daily_usd(self): return self.state["daily"].get(dt.date.today().isoformat(), 0.0)

    # ---- discovery ------------------------------------------------------------------
    def discover(self):
        ev = []; off = 0
        while True:
            page = get_json(f"{C.GAMMA}/events", {"tag_id": 84, "closed": "false", "limit": 100, "offset": off})
            if not page: break
            ev += page; off += 100
        mk = {}
        for e in ev:
            t = TITLE_RE.match(e["title"])
            if not t or t.group(1) not in TZ: continue
            city = t.group(1); desc = e["markets"][0].get("description", ""); d = re.search(r"on (\d{1,2} \w{3} '\d\d)", desc)
            if not d: continue
            day = dt.datetime.strptime(d.group(1), "%d %b '%y").date()
            for m in e["markets"]:
                toks = json.loads(m.get("clobTokenIds") or "[]")
                if len(toks) != 2 or m.get("closed"): continue
                mk[m["conditionId"]] = dict(cid=m["conditionId"], city=city, tz=ZoneInfo(TZ[city]), day=day, question=m["question"], yes_token=toks[0], no_token=toks[1])
        self.markets = mk; log.info("discovery: %d buckets", len(mk))

    def in_window(self, m):
        now = dt.datetime.now(m["tz"]); return now.date() == m["day"] and C.WINDOW[0] <= now.hour < C.WINDOW[1]
    def past_window(self, m):
        now = dt.datetime.now(m["tz"]); return now.date() > m["day"] or (now.date() == m["day"] and now.hour >= C.WINDOW[1])

    # ---- touch test ------------------------------------------------------------------
    def has_touched(self, m):
        if m["cid"] in self.touched: return self.touched[m["cid"]]
        t0 = int(dt.datetime(m["day"].year, m["day"].month, m["day"].day, tzinfo=m["tz"]).timestamp())
        try:
            h = get_json(f"{C.CLOB}/prices-history", {"market": m["yes_token"], "startTs": t0, "endTs": int(time.time()), "fidelity": 1}).get("history", [])
            mx = max((x["p"] for x in h), default=0.0)
        except Exception as ex:
            log.warning("prices-history %s: %s", m["question"][:40], ex); return False
        self.touched[m["cid"]] = mx >= C.TOUCH
        return self.touched[m["cid"]]

    # ---- ladder placement / cancel -------------------------------------------------------
    def place_ladder(self, m):
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        orders = {}
        opts = None
        if not C.DRY_RUN:
            opts = PartialCreateOrderOptions(tick_size=self.client.get_tick_size(m["yes_token"]), neg_risk=self.client.get_neg_risk(m["yes_token"]))
        for lvl in C.LEVELS:
            shares = max(C.MIN_ORDER_SHARES, round(C.STAKE_PER_LEVEL / lvl, 2))
            if C.DRY_RUN:
                orders[str(lvl)] = dict(id=f"dry-{m['cid'][:8]}-{lvl}", shares=shares, filled=0.0)
            else:
                try:
                    o = self.client.create_order(OrderArgs(token_id=m["yes_token"], price=lvl, size=shares, side="BUY"), opts)
                    r = self.client.post_order(o, OrderType.GTC); orders[str(lvl)] = dict(id=r.get("orderID") or r.get("id"), shares=shares, filled=0.0)
                except Exception as ex:
                    log.error("order %s @ %.2f failed: %s", m["question"][:40], lvl, ex)
        self.state["ladders"][m["cid"]] = dict(question=m["question"], city=m["city"], placed=time.time(), orders=orders, yes_token=m["yes_token"], no_token=m["no_token"], open=True)
        self.save()
        log.info("LADDER UP: %s | %d levels %.2f..%.2f, %s shares each [%s]", m["question"], len(orders), C.LEVELS[0], C.LEVELS[-1],
                 ", ".join(str(o["shares"]) for o in list(orders.values())[:3]) + "..", "DRY" if C.DRY_RUN else "LIVE")

    def cancel_ladder(self, cid, why):
        L = self.state["ladders"][cid]
        if not C.DRY_RUN:
            for o in L["orders"].values():
                try: self.client.cancel(o["id"])
                except Exception as ex: log.warning("cancel %s: %s", o["id"], ex)
        L["open"] = False; self.save()
        filled = sum(o["filled"] for o in L["orders"].values())
        log.info("LADDER DOWN (%s): %s | filled %.2f shares", why, L["question"], filled)

    # ---- fill tracking -----------------------------------------------------------------
    def check_fills(self, cid):
        L = self.state["ladders"][cid]
        if C.DRY_RUN:
            # paper fills from the public tape: taker SELL YES <= level, or taker BUY NO >= 1-level, after placement
            try: tape = get_json(f"{C.DATA_API}/trades", {"market": cid, "limit": 200})
            except Exception as ex: log.warning("tape %s: %s", cid[:8], ex); return
            for lvl_s, o in L["orders"].items():
                if o["filled"] >= o["shares"]: continue
                lvl = float(lvl_s)
                hits = [t for t in tape if t["timestamp"] > L["placed"] and
                        ((str(t["asset"]) == L["yes_token"] and t["side"] == "SELL" and float(t["price"]) <= lvl) or
                         (str(t["asset"]) == L["no_token"] and t["side"] == "BUY" and 1 - float(t["price"]) <= lvl))]
                if hits:
                    self._record_fill(cid, lvl, o["shares"] - o["filled"], o); o["filled"] = o["shares"]
        else:
            for lvl_s, o in L["orders"].items():
                if o["filled"] >= o["shares"] or not o.get("id"): continue
                try: info = self.client.get_order(o["id"])
                except Exception: continue
                matched = float(info.get("size_matched") or 0)
                if matched > o["filled"]:
                    self._record_fill(cid, float(lvl_s), matched - o["filled"], o); o["filled"] = matched
        self.save()

    def _record_fill(self, cid, lvl, shares, o):
        L = self.state["ladders"][cid]; usd = shares * lvl; today = dt.date.today().isoformat()
        p = self.state["positions"].setdefault(cid, {"question": L["question"], "shares": 0.0, "usd": 0.0}); p["shares"] += shares; p["usd"] += usd
        self.state["daily"][today] = self.daily_usd() + usd
        new = not os.path.exists(C.TRADE_LOG)
        with open(C.TRADE_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["time_utc", "city", "question", "level", "shares", "usd", "dry_run"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), L["city"], L["question"], lvl, round(shares, 2), round(usd, 2), C.DRY_RUN])
        log.info("FILL: %s @ %.2f x %.2f = $%.2f [%s]", L["question"], lvl, shares, usd, "DRY" if C.DRY_RUN else "LIVE")

    # ---- main loop ----------------------------------------------------------------------
    def run(self):
        while True:
            try:
                now = time.time()
                if now - self.t_disc > C.DISCOVERY_SECONDS: self.discover(); self.t_disc = now
                halted = os.path.exists(C.KILL_FILE) or self.daily_usd() >= C.MAX_DAILY_USD
                open_ladders = [c for c, L in self.state["ladders"].items() if L["open"]]
                # new ladders
                for cid, m in self.markets.items():
                    if cid in self.state["ladders"] or not self.in_window(m) or halted or len(open_ladders) >= C.MAX_MARKETS_OPEN: continue
                    # cheap prefilter: only buckets whose YES is near the top now
                    try: mid = float(get_json(f"{C.CLOB}/midpoint", {"token_id": m["yes_token"]}).get("mid", 0))
                    except Exception: continue
                    if mid < 0.90: continue
                    if self.has_touched(m):
                        self.place_ladder(m); open_ladders.append(cid)
                    time.sleep(0.05)
                # maintain open ladders
                for cid in open_ladders:
                    m = self.markets.get(cid); self.check_fills(cid)
                    if m is None or self.past_window(m) or halted: self.cancel_ladder(cid, "window end" if m else "market gone")
            except Exception as ex:
                log.exception("loop error: %s", ex)
            time.sleep(C.POLL_SECONDS)

if __name__ == "__main__":
    Ladder().run()
