"""Polymarket weather bot: buy NO on a temperature bucket the moment an hourly
METAR print rules it out.

Loop (see config.py for the knobs):
  every DISCOVERY_SECONDS   pull today's/tomorrow's "Highest temperature in <city>" events
                            from Gamma; parse station, unit, day, buckets, NO token ids.
  every BOOK_REFRESH_SECONDS for cities inside their local 11:00-18:00 window, read each
                            bucket's NO best ask; buckets with ask <= MAX_ASK and room left
                            under MAX_POSITION_USD are "candidates".
  every POLL_SECONDS        one METAR request for all active stations. For each NEW
                            observation: rounded temp (T-group tenths -> unit, half-up).
                            If it exceeds a candidate bucket's upper bound -> execute().
  execute()                 read the NO book; take the first ASK_LEVELS levels that are
                            <= MAX_ASK; size = min(their total, room left); place a FAK
                            limit at the worst of those levels; re-read the book; repeat
                            until no ask <= MAX_ASK, room is exhausted, or MAX_FILL_LOOPS.
                            Then back to polling.

Run:  BOT_DRY_RUN=1 python3 bot/metar_no_bot.py        (default; logs, never orders)
      BOT_DRY_RUN=0 POLY_PRIVATE_KEY=... python3 bot/metar_no_bot.py
"""
import json, os, re, sys, time, csv, logging, datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import config as C
from sweep_local import TZ
TZ = {**TZ, "NYC": "America/New_York", "New York": "America/New_York", "Seoul (Incheon)": "Asia/Seoul"}
for _n in ("httpx", "httpcore", "urllib3"): logging.getLogger(_n).setLevel(logging.WARNING)

log = logging.getLogger("bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("bot/bot.log")])

BUCKET_RE = re.compile(r"be (?:between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC])")
TITLE_RE = re.compile(r"^Highest temperature in (.+?) on ")
TG_RE = re.compile(r"\sT(\d)(\d{3})(\d)(\d{3})")

def parse_bucket(q):
    m = BUCKET_RE.search(q)
    if not m: return None
    if m.group(1): return (int(m.group(1)), int(m.group(2)))
    if m.group(3): return None if m.group(4) == "higher" else (-999, int(m.group(3)))  # "or higher" can't be ruled out by heat
    return (int(m.group(5)), int(m.group(5)))

def round_half_up(x):
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def obs_temp(o, unit):
    """Rounded temperature in the market's unit from a METAR JSON record.
    Uses the T-group tenths when present (that is what NOAA's WRH page shows)."""
    tg = TG_RE.search(o.get("rawOb", "") or "")
    c = (-1 if tg.group(1) == "1" else 1) * int(tg.group(2)) / 10 if tg else o.get("temp")
    if c is None: return None
    return round_half_up(c * 9 / 5 + 32) if unit == "F" else round_half_up(c)

def get_json(url, params, headers=None, tries=3, backoff=2.0, timeout=20):
    """GET + JSON with a quick retry: feeds return empty bodies / timeouts at the top of the hour,
    exactly when the new prints land, so waiting for the next poll cycle is the wrong response."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            last = f"HTTP {r.status_code}, {len(r.text)} bytes"
        except Exception as ex:
            last = repr(ex)[:120]
        time.sleep(backoff)
    raise RuntimeError(f"{url.split('/')[2]}: {last} after {tries} tries")

# ---------------------------------------------------------------------------------
class Bot:
    def __init__(self):
        self.markets = {}        # station -> dict(city, unit, tz, day, buckets=[{...}])
        self.candidates = {}     # no_token -> bucket dict (ask <= MAX_ASK and room left)
        self.seen_obs = set()    # (source, station, obs_time)
        self.running_max = {}    # (station, local_day, kind) -> int   kind: '5min' | 'hourly'
        self.handled = set()     # (cid, kind): a bucket is acted on once per trigger kind
        self.t_hourly = 0
        self.state = self._load_state()
        self.client = self._make_client()
        self.t_disc = self.t_book = 0

    # ---- state / client -------------------------------------------------------------
    def _load_state(self):
        if os.path.exists(C.STATE_FILE):
            return json.load(open(C.STATE_FILE))
        return {"positions": {}, "daily": {}}
    def _save_state(self):
        json.dump(self.state, open(C.STATE_FILE + ".tmp", "w"), indent=1); os.replace(C.STATE_FILE + ".tmp", C.STATE_FILE)
    def _make_client(self):
        from py_clob_client.client import ClobClient
        if C.DRY_RUN or not C.PRIVATE_KEY:
            log.info("DRY RUN: no orders will be placed")
            return ClobClient(C.CLOB)
        c = ClobClient(C.CLOB, key=C.PRIVATE_KEY, chain_id=C.CHAIN_ID, signature_type=C.SIGNATURE_TYPE, funder=C.FUNDER)
        c.set_api_creds(c.create_or_derive_api_creds())
        log.info("LIVE: trading enabled for %s", c.get_address())
        return c
    def position_usd(self, cid):  return self.state["positions"].get(cid, {}).get("usd", 0.0)
    def confirmed(self, cid):     return self.state["positions"].get(cid, {}).get("confirmed", False)
    def cap_usd(self, b):         return C.MAX_POSITION_USD * (1.0 if b.get("confirmed") else C.TRANCHE_5MIN)
    def daily_usd(self):
        today = dt.date.today().isoformat(); return self.state["daily"].get(today, 0.0)
    def trading_allowed(self):
        if os.path.exists(C.KILL_FILE): return False
        return self.daily_usd() < C.MAX_DAILY_USD

    # ---- discovery ----------------------------------------------------------------
    def discover(self):
        events = []; off = 0
        while True:
            page = requests.get(f"{C.GAMMA}/events", params={"tag_id": 84, "closed": "false", "limit": 100, "offset": off}, timeout=30).json()
            if not page: break
            events += page; off += 100
        markets = {}
        for e in events:
            t = TITLE_RE.match(e["title"])
            if not t: continue
            city = t.group(1); tzname = TZ.get(city)
            if not tzname: log.warning("no timezone for city %r, skipping", city); continue
            desc = e["markets"][0].get("description", "")
            site = re.search(r"site=(\w+)", desc); u = re.search(r"in degrees (Celsius|Fahrenheit)", desc); d = re.search(r"on (\d{1,2} \w{3} '\d\d)", desc)
            if not (site and u and d): continue
            station = site.group(1).upper(); unit = u.group(1)[0]; day = dt.datetime.strptime(d.group(1), "%d %b '%y").date()
            buckets = []
            for m in e["markets"]:
                b = parse_bucket(m["question"]); toks = json.loads(m.get("clobTokenIds") or "[]")
                if not b or len(toks) != 2 or m.get("closed"): continue
                buckets.append(dict(question=m["question"], hi=b[1], lo=b[0], no_token=toks[1], cid=m["conditionId"], city=city, station=station, day=day,
                                    confirmed=self.confirmed(m["conditionId"])))
            key = (station, day)
            markets[key] = dict(city=city, unit=unit, tz=ZoneInfo(tzname), day=day, buckets=buckets)
        self.markets = markets
        log.info("discovery: %d station-days, %d buckets", len(markets), sum(len(v["buckets"]) for v in markets.values()))

    def active(self):
        """station-day entries whose local date is today and local hour in the window."""
        out = {}
        for (station, day), m in self.markets.items():
            now = dt.datetime.now(m["tz"])
            if now.date() == day and C.LOCAL_WINDOW[0] <= now.hour < C.LOCAL_WINDOW[1]:
                out[station] = m
        return out

    # ---- books --------------------------------------------------------------------
    def best_ask(self, token):
        try:
            ob = self.client.get_order_book(token)
            asks = sorted(((float(a.price), float(a.size)) for a in ob.asks), key=lambda x: x[0])
            return asks
        except Exception as ex:
            log.warning("book error %s: %s", token[:12], ex); return None

    def refresh_candidates(self):
        cands = {}
        for station, m in self.active().items():
            for b in m["buckets"]:
                if self.position_usd(b["cid"]) >= C.MAX_POSITION_USD: continue
                asks = self.best_ask(b["no_token"])
                if asks and asks[0][0] <= C.MAX_ASK:
                    b["asks"] = asks; cands[b["no_token"]] = b
                time.sleep(0.05)
        self.candidates = cands
        log.info("candidates: %d buckets with NO ask <= %.2f in %d active cities: %s", len(cands), C.MAX_ASK, len(self.active()),
                 ", ".join(f"{b['city']} {b['question'].split(' be ',1)[1].split(' on ')[0]}@{b['asks'][0][0]}" for b in list(cands.values())[:12]))

    # ---- signals --------------------------------------------------------------------
    def on_reading(self, st, m, temp, t_obs, kind, source):
        """A new observation for station st (market m). kind: '5min' or 'hourly'."""
        if m["unit"] == "C": kind = "hourly"          # °C stations: both feeds carry the same half-hourly METAR
        tl = t_obs.astimezone(m["tz"])
        if tl.date() != m["day"]: return
        age_min = (dt.datetime.now(dt.timezone.utc) - t_obs).total_seconds() / 60
        rk = (st, m["day"], kind); prev = self.running_max.get(rk, -999)
        if temp <= prev: return
        self.running_max[rk] = temp
        margin = 0 if kind == "hourly" else (C.MARGIN5_F if m["unit"] == "F" else C.MARGIN5_C) - 1
        log.info("%s %s %s new high %d°%s at %s local via %s (age %.0f min)", m["city"], st, kind, temp, m["unit"], tl.strftime("%H:%M"), source, age_min)
        for b in m["buckets"]:
            hi = b["hi"]
            if not (prev - margin <= hi < temp - margin): continue          # newly cleared by this reading (with margin)
            if (b["cid"], kind) in self.handled: continue                  # same reading seen via the other feed
            self.handled.add((b["cid"], kind))
            if kind == "hourly" or m["unit"] == "C":
                b["confirmed"] = True; self._mark_confirmed(b)
            if self.position_usd(b["cid"]) >= self.cap_usd(b): continue
            if age_min > C.MAX_PRINT_AGE_MIN:
                log.info("skip stale %s reading (%.0f min): %s", kind, age_min, b["question"]); continue
            asks = self.best_ask(b["no_token"]) or []
            pre = (b.get("asks") or [(None, None)])[0][0]      # last cached best ask before this print (candidates refresh)
            log.info("%s: %s  (%s reading %d vs bucket hi %d, margin %d) | NO ask before print %s | NO book top5 asks now %s", "CONFIRMED" if b.get("confirmed") else "TRIGGER", b["question"], kind, temp, hi, margin + 1, pre, asks[:5])
            self.execute(b, temp)
        # contradiction: an hourly print that fails to clear a bucket we already bought on a 5-min trigger
        if kind == "hourly":
            for b in m["buckets"]:
                if self.position_usd(b["cid"]) > 0 and not self.confirmed(b["cid"]) and temp <= b["hi"]:
                    log.warning("CONTRADICTION: hourly print %d does not clear %s (hi %d); position $%.2f unconfirmed", temp, b["question"], b["hi"], self.position_usd(b["cid"]))

    def _mark_confirmed(self, b):
        p = self.state["positions"].get(b["cid"])
        if p and not p.get("confirmed"):
            p["confirmed"] = True; self._save_state(); log.info("confirmed by hourly print: %s", b["question"])

    def poll_synoptic(self):
        act = self.active()
        if not act: return
        try:
            r = get_json(C.SYNOPTIC, {"STID": ",".join(act), "recent": 90, "token": C.SYNOPTIC_TOKEN, "obtimezone": "utc"}, headers=C.SYNOPTIC_HEADERS)
        except Exception as ex:
            log.warning("synoptic error: %s", ex); return
        if not r.get("STATION"):
            log.warning("synoptic: %s", r.get("SUMMARY", {}).get("RESPONSE_MESSAGE")); return
        for s in r["STATION"]:
            st = s["STID"].upper(); m = act.get(st)
            if not m: continue
            o = s.get("OBSERVATIONS", {}); temps = o.get("air_temp_set_1") or []
            for tstr, c in zip(o.get("date_time", []), temps):
                if c is None: continue
                key = ("syn", st, tstr)
                if key in self.seen_obs: continue
                self.seen_obs.add(key)
                t_obs = dt.datetime.fromisoformat(tstr.replace("Z", "+00:00"))
                temp = round_half_up(c * 9 / 5 + 32) if m["unit"] == "F" else round_half_up(c)
                # a reading carrying tenths is the hourly METAR itself (5-min ASOS reports are whole °C)
                kind = "hourly" if (abs(c * 10 - round(c * 10)) < 1e-6 and round(c * 10) % 10 != 0) else "5min"
                self.on_reading(st, m, temp, t_obs, kind, "synoptic")

    def poll_hourly(self, hours=2):
        act = self.active()
        if not act: return
        try:
            obs = get_json(C.METAR, {"ids": ",".join(act), "format": "json", "hours": hours})
        except Exception as ex:
            log.warning("metar error: %s", ex); return
        for o in sorted(obs, key=lambda o: o.get("obsTime", 0)):
            st = o["icaoId"]; m = act.get(st)
            if not m or not o.get("obsTime"): continue
            key = ("avwx", st, o["obsTime"])
            if key in self.seen_obs: continue
            self.seen_obs.add(key)
            temp = obs_temp(o, m["unit"])
            if temp is None: continue
            t_obs = dt.datetime.fromtimestamp(o["obsTime"], dt.timezone.utc)
            self.on_reading(st, m, temp, t_obs, "hourly", "aviationweather")

    # ---- execution ----------------------------------------------------------------
    def execute(self, b, temp):
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        for i in range(C.MAX_FILL_LOOPS):
            if not self.trading_allowed():
                log.warning("trading disabled (kill file or daily cap) — not buying %s", b["question"]); return
            cap = self.cap_usd(b); room = cap - self.position_usd(b["cid"])
            if room <= 0: log.info("position cap $%.0f reached for %s (%s)", cap, b["question"], "confirmed" if b.get("confirmed") else "5-min tranche"); break
            asks = self.best_ask(b["no_token"]) or []
            levels = [a for a in asks if a[0] <= C.MAX_ASK][:C.ASK_LEVELS]
            if not levels: log.info("no NO ask <= %.2f left for %s (best %s)", C.MAX_ASK, b["question"], asks[:1]); break
            price = max(p for p, _ in levels)
            shares = min(sum(s for _, s in levels), room / price)
            shares = float(int(shares * 100) / 100)
            if shares < C.MIN_ORDER_SHARES: log.info("size %.2f below minimum, stop", shares); break
            log.info("BUY NO %s: %.2f shares @ %.3f (levels %s, room $%.2f) [%s]", b["question"], shares, price, levels, room, "DRY" if C.DRY_RUN else "LIVE")
            filled = shares if C.DRY_RUN else self._place(b["no_token"], price, shares, OrderArgs, OrderType, PartialCreateOrderOptions)
            if filled <= 0: log.info("nothing filled, stop"); break
            self._record(b, price, filled, temp)
            time.sleep(0.5)
            if C.DRY_RUN: break      # in dry run the book doesn't change; don't pretend to fill repeatedly
        if self.position_usd(b["cid"]) >= C.MAX_POSITION_USD: self.candidates.pop(b["no_token"], None)

    def _place(self, token, price, shares, OrderArgs, OrderType, PartialCreateOrderOptions):
        try:
            opts = PartialCreateOrderOptions(tick_size=self.client.get_tick_size(token), neg_risk=self.client.get_neg_risk(token))
            order = self.client.create_order(OrderArgs(token_id=token, price=price, size=shares, side="BUY"), opts)
            resp = self.client.post_order(order, OrderType.FAK)     # fill what is there at <= price, cancel the rest
            oid = resp.get("orderID") or resp.get("id")
            time.sleep(0.3)
            info = self.client.get_order(oid) if oid else {}
            filled = float(info.get("size_matched") or 0)
            log.info("order %s status=%s filled=%.2f", oid, info.get("status"), filled)
            return filled
        except Exception as ex:
            log.error("order failed: %s", ex); return 0.0

    def _record(self, b, price, shares, temp):
        usd = price * shares; today = dt.date.today().isoformat()
        p = self.state["positions"].setdefault(b["cid"], {"question": b["question"], "shares": 0.0, "usd": 0.0, "confirmed": False})
        p["shares"] += shares; p["usd"] += usd; p["confirmed"] = p["confirmed"] or bool(b.get("confirmed"))
        self.state["daily"][today] = self.state["daily"].get(today, 0.0) + usd
        self._save_state()
        new = not os.path.exists(C.TRADE_LOG)
        with open(C.TRADE_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new: w.writerow(["time_utc", "city", "station", "question", "reading", "bucket_hi", "price", "shares", "usd", "dry_run", "trigger"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), b["city"], b["station"], b["question"], temp, b["hi"], price, shares, round(usd, 2), C.DRY_RUN, "confirmed" if b.get("confirmed") else "5min"])

    # ---- main loop ------------------------------------------------------------------
    def run(self):
        first = True
        while True:
            try:
                now = time.time()
                if now - self.t_disc > C.DISCOVERY_SECONDS: self.discover(); self.t_disc = now
                if now - self.t_book > C.BOOK_REFRESH_SECONDS: self.refresh_candidates(); self.t_book = now
                if first or now - self.t_hourly > C.HOURLY_POLL_SECONDS:
                    self.poll_hourly(hours=24 if first else 2)   # first poll seeds today's hourly running max
                    self.t_hourly = now
                self.poll_synoptic()                              # primary trigger, every POLL_SECONDS
                first = False
            except Exception as ex:
                log.exception("loop error: %s", ex)
            time.sleep(C.POLL_SECONDS)

if __name__ == "__main__":
    Bot().run()
