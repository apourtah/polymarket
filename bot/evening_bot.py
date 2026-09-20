"""Evening HRRR bot: EWMA-corrected 00Z HRRR + ridge regime model, per-bucket probabilities, YES only.

Nightly, per city, between 21:00 and 23:00 local (the window where the market has not yet absorbed the
latest short-range run):
  1. features for TOMORROW: latest extended HRRR run (00/06/12/18Z, from AWS via herbie) afternoon max at the
     station; HRRR weather fields from Open-Meteo (gfs_hrrr); today's/yesterday's METAR max/min; NBM + GFS-MOS
     station maxes from IEM; the station's recent error history.
  2. models refit on the history panel: EWMA per station (gain tuned), pooled ridge on FEATS.
  3. P(bucket) for each of tomorrow's buckets under each model (normal with that model's residual sd).
  4. agreement (same best bucket): buy it if ask <= AGREE_MAX_PRICE.
     disagreement: buy every bucket where avg(P) - ask >= DISAGREE_EDGE.
  5. FAK at the ask, $STAKE (x2 if edge >= 2*DISAGREE_EDGE), caps per market / city-day / day.
  Each morning-after: score yesterday's forecast against the METAR max and append to history.
Run: BOT_DRY_RUN=1 python3 bot/evening_bot.py   |   BOT_DRY_RUN=0 OWNER_PRIVATE_KEY=0x... POLYMARKET_PROXY=0x... POLYMARKET_LOGIN=email|metamask python3 bot/evening_bot.py
"""
import json, os, re, sys, csv, io, time, logging, datetime as dt, warnings
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd, requests
from scipy.stats import norm
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import evening_config as C
from sweep_local import TZ
warnings.filterwarnings("ignore")
log = logging.getLogger("evening"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(C.LOG_FILE)])
for n in ("httpx", "httpcore", "urllib3", "herbie"): logging.getLogger(n).setLevel(logging.WARNING)
rh = lambda x: int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
SITE = json.load(open("out/stations.json")); COORDS = json.load(open("out/station_coords.json"))
BUCKET_RE = re.compile(r"be (?:between (-?\d+)-(-?\d+)|(-?\d+)°F or (higher|below)|(-?\d+)°F on)")
def parse_bucket(q):
    m = BUCKET_RE.search(q)
    if not m: return None
    if m.group(1): return (int(m.group(1)), int(m.group(2)))
    if m.group(3): return (int(m.group(3)), 999) if m.group(4) == "higher" else (-999, int(m.group(3)))
    return (int(m.group(5)), int(m.group(5)))
def get_json(url, params=None, headers=None, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200 and r.text.strip(): return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as ex: last = repr(ex)[:80]
        time.sleep(2)
    raise RuntimeError(f"{url.split('/')[2]}: {last}")

# ---------------------------------------------------------------- data sources
_grid_idx = {}
def hrrr_afternoon_max(city, target_day, tz):
    """Afternoon (11-18 local) max 2-m temp at the station from the 00Z HRRR run of the target day's UTC date
    (the run the history was built from). Returns (None, run) until that run's forecast hours are all available."""
    from herbie import Herbie
    st = SITE[city][0]; lat, lon = COORDS[st]
    run = dt.datetime(target_day.year, target_day.month, target_day.day, 0, tzinfo=dt.timezone.utc)   # 00Z on the target date
    vals = []
    for lh in range(11, 19):
        vt = dt.datetime(target_day.year, target_day.month, target_day.day, lh, tzinfo=tz).astimezone(dt.timezone.utc)
        fxx = int((vt - run).total_seconds() // 3600)
        try:
            H = Herbie(run.strftime("%Y-%m-%d %H:00"), model="hrrr", product="sfc", fxx=fxx, verbose=False)
            if not H.grib: return None, run                              # this forecast hour not published yet
            ds = H.xarray("TMP:2 m above ground", remove_grib=True)
            if st not in _grid_idx:
                la = ds.latitude.values; lo_ = ds.longitude.values - 360; d = (la - lat) ** 2 + (lo_ - lon) ** 2; _grid_idx[st] = np.unravel_index(d.argmin(), d.shape)
            vals.append(float(ds.t2m.values[_grid_idx[st]]) * 9 / 5 - 459.67)
        except Exception as ex:
            log.info("HRRR %s 00Z f%02d not available yet", st, fxx); return None, run
    return max(vals), run

def openmeteo_fields(city, target_day, tz):
    """Surface fields for the target afternoon from the Open-Meteo PREVIOUS-RUNS API (`*_previous_day1` = the run issued the
    day before the valid hour), i.e. exactly what the history/backtest rows were built from (data/hrrr_wx_d1.parquet). At 21:00
    local the day before, those runs all exist. Falls back to the current-run forecast API if the archive call fails."""
    st = SITE[city][0]; lat, lon = COORDS[st]
    V = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "cloud_cover", "shortwave_radiation", "surface_pressure", "precipitation"]
    j = get_json("https://previous-runs-api.open-meteo.com/v1/forecast", {"latitude": lat, "longitude": lon, "hourly": ",".join(v + "_previous_day1" for v in V), "start_date": target_day.isoformat(), "end_date": target_day.isoformat(),
                                                                          "models": "gfs_hrrr", "timezone": tz.key, "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"})
    if j and "hourly" in j and any(x is not None for x in j["hourly"].get("temperature_2m_previous_day1", [])):
        h = pd.DataFrame({"time": pd.to_datetime(j["hourly"]["time"]), **{v: j["hourly"][v + "_previous_day1"] for v in V}})
    else:
        log.warning("%s: previous-runs API unavailable, using the current-run forecast for the surface fields", city)
        j = get_json(C.OPEN_METEO, {"latitude": lat, "longitude": lon, "hourly": ",".join(V), "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "models": "gfs_hrrr", "timezone": tz.key, "forecast_days": 3})
        h = pd.DataFrame(j["hourly"]); h["time"] = pd.to_datetime(h.time)
    h = h[h.time.dt.date == target_day]
    a = h[h.time.dt.hour.between(11, 18)]; m = h[h.time.dt.hour.between(4, 8)]
    wd = np.degrees(np.arctan2(np.sin(np.radians(a.wind_direction_10m)).mean(), np.cos(np.radians(a.wind_direction_10m)).mean())) % 360
    return dict(dew=a.dew_point_2m.mean(), rhum=a.relative_humidity_2m.mean(), wind=a.wind_speed_10m.mean(), cloud=a.cloud_cover.mean(), rad=a.shortwave_radiation.mean(), pres=a.surface_pressure.mean(), precip=a.precipitation.sum(),
                wdir_s=np.sin(np.radians(wd)), wdir_c=np.cos(np.radians(wd)), t_morn=m.temperature_2m.mean(), cloud_morn=m.cloud_cover.mean())

def metar_day_max_min(city, day, tz, hours=48):
    st = SITE[city][0]; obs = get_json(C.METAR, {"ids": st, "format": "json", "hours": hours})
    vals = [rh(o["temp"] * 9 / 5 + 32) for o in obs if o.get("temp") is not None and o.get("obsTime") and dt.datetime.fromtimestamp(o["obsTime"], tz).date() == day and (o.get("metarType") == "METAR" or True)]
    return (max(vals), min(vals)) if vals else (np.nan, np.nan)

def mos_max(city, model, target_day, tz):
    """Afternoon max from the latest 00Z (IEM stores older NBS runs as 01Z) MOS run issued before 21:00 local the day before —
    the same run selection as the history/backtest rows (backtest_evening.py mos_daily)."""
    st = SITE[city][0]; now = dt.datetime.now(dt.timezone.utc)
    r = requests.get(C.IEM_MOS, params={"station": st, "model": model, "sts": (now - dt.timedelta(hours=54)).strftime("%Y-%m-%dT%H:00Z"), "ets": now.strftime("%Y-%m-%dT%H:00Z"), "format": "csv"}, timeout=60)
    if r.status_code != 200 or len(r.text) < 100: return np.nan
    d = pd.read_csv(io.StringIO(r.text)); d["ftime"] = pd.to_datetime(d.ftime, utc=True); d["runtime"] = pd.to_datetime(d.runtime, utc=True)
    cutoff = pd.Timestamp(dt.datetime(target_day.year, target_day.month, target_day.day, 21, tzinfo=tz) - dt.timedelta(days=1))
    d = d[d.runtime.dt.hour.isin([0, 1]) & (d.runtime <= cutoff)]
    if d.empty: return np.nan
    d = d[d.runtime == d.runtime.max()]; lt = d.ftime.dt.tz_convert(tz); d = d[(lt.dt.date == target_day) & lt.dt.hour.between(11, 18)]
    return pd.to_numeric(d.tmp, errors="coerce").max() if len(d) else np.nan

# ---------------------------------------------------------------- models
class Models:
    def __init__(self, hist):
        self.hist = hist.dropna(subset=["err"]).copy(); self.cities = sorted(self.hist.city.unique())
        self.ewma = {}; self.ewma_sd = {}; self.ridge_sd = {}
        for c in self.cities:
            e = self.hist[self.hist.city == c].sort_values("mday").err.values
            best = None
            for k in C.EWMA_GAINS:
                b = 0.0; se = []; 
                for x in e: se.append(x - b); b += k * (x - b)
                s = float(np.sum(np.square(se)))
                if best is None or s < best[0]: best = (s, k, b, np.std(se[-60:]))
            self.ewma[c] = (best[1], best[2]); self.ewma_sd[c] = max(best[3], 1.0)
        X = self.hist[C.FEATS].copy(); self.med = X.median(); X = X.fillna(self.med); self.mu = X.mean(); self.sd = X.std().replace(0, 1)
        Z = np.c_[((X - self.mu) / self.sd).values, pd.get_dummies(self.hist.city).reindex(columns=self.cities, fill_value=0).values]
        self.ridge = Ridge(alpha=C.RIDGE_ALPHA).fit(Z, self.hist.err.values)
        res = self.hist.err.values - self.ridge.predict(Z)
        for c in self.cities: self.ridge_sd[c] = max(float(np.std(res[(self.hist.city == c).values][-60:])), 1.0)
    def predict(self, city, feats):
        x = pd.Series({f: feats.get(f, np.nan) for f in C.FEATS}).fillna(self.med)
        z = np.r_[((x - self.mu) / self.sd).values, [1.0 if c == city else 0.0 for c in self.cities]]
        return dict(ewma=self.ewma[city][1], ridge=float(self.ridge.predict(z[None, :])[0]), ewma_sd=self.ewma_sd[city], ridge_sd=self.ridge_sd[city])
def bucket_probs(mu, sd, buckets):
    return {b: norm.cdf((min(b[1], 200) + 0.5 - mu) / sd) - norm.cdf((max(b[0], -200) - 0.5 - mu) / sd) for b in buckets}

def window_pos(now):
    """Minutes since the window opened at WINDOW[0] local, or None if outside. The window may cross midnight
    (WINDOW = (21, 25) means 21:00 -> 01:00 next day); the target market day is always the day after the 21:00 opening."""
    start, end = C.WINDOW; h = now.hour + now.minute / 60
    if start <= h < min(end, 24): return int((h - start) * 60), (now + dt.timedelta(days=1)).date()
    if end > 24 and h < end - 24: return int((h + 24 - start) * 60), now.date()
    return None

# ---------------------------------------------------------------- bot
class Bot:
    def __init__(self):
        self.state = json.load(open(C.STATE_FILE)) if os.path.exists(C.STATE_FILE) else {"done": {}, "positions": {}, "daily": {}}
        self.client = self._client()
    def _client(self):
        """Read-only client; trading clients are created per wallet in `wallet_client`."""
        from py_clob_client.client import ClobClient
        if C.DRY_RUN or not C.WALLETS[0].get("key"): log.info("DRY RUN: no orders will be placed (%d wallet slot(s))", len(C.WALLETS))
        else: log.info("LIVE: %d wallet(s)", len(C.WALLETS))
        self._wc = {}; self._redeem_ts = {}; return ClobClient(C.CLOB)
    def pick_wallet(self, city, target, why=None):
        n = len(C.WALLETS); mode = C.WALLET_MODE
        if n == 1: return 0
        if mode == "per_city": return int(C.WALLET_CITY_MAP.get(city, sum(map(ord, city)))) % n
        if mode == "per_signal": return int(C.WALLET_SIGNAL_MAP.get(why or "agree", 0)) % n
        if mode == "round_robin":
            k = self.state.setdefault("rr", 0); self.state["rr"] = k + 1; return k % n
        if mode == "per_day": return (target.toordinal() + self.state.setdefault("day_salt", __import__("random").randrange(n))) % n
        return __import__("random").randrange(n)                       # random
    def wallet_client(self, i):
        from py_clob_client.client import ClobClient
        if i not in self._wc:
            w = C.WALLETS[i]; c = ClobClient(C.CLOB, key=w["key"], chain_id=C.CHAIN_ID, signature_type=w.get("sig_type", 0), funder=w.get("funder")); c.set_api_creds(c.create_or_derive_api_creds()); self._wc[i] = c
        return self._wc[i]
    def ensure_cash(self, wi, need):
        """True if wallet `wi` can fund `need` $. If its free USDC is short, redeem resolved winners first (at most once per 15 min per wallet)."""
        w = C.WALLETS[wi]
        if not C.AUTO_REDEEM or not w.get("key"): return True
        import bot.redeem as R
        cash = R.free_cash(self.wallet_client(wi))
        if cash is None or cash >= need + C.CASH_RESERVE: return True
        if time.time() - self._redeem_ts.get(wi, 0) < 900: return cash >= need
        self._redeem_ts[wi] = time.time(); log.info("wallet #%d: free cash $%.2f < $%.2f needed (+$%.0f reserve) -> redeeming resolved winners", wi, cash, need, C.CASH_RESERVE)
        n, usd = R.redeem_all(w, dry_run=C.DRY_RUN)
        if n and not C.DRY_RUN: time.sleep(5); cash = R.free_cash(self.wallet_client(wi)) or cash
        return cash >= need
    def save(self): json.dump(self.state, open(C.STATE_FILE + ".tmp", "w"), indent=1); os.replace(C.STATE_FILE + ".tmp", C.STATE_FILE)
    def daily_usd(self): return self.state["daily"].get(dt.date.today().isoformat(), 0.0)

    # ---- history maintenance ----
    def history(self): return pd.read_parquet(C.HISTORY) if os.path.exists(C.HISTORY) else pd.DataFrame()
    def score_pending(self):
        """Fill in `actual`/`err` for the bot's own forecasts. A day is scored PROVISIONALLY from 21:00 local on the day itself
        (the max is set by then; this is what the backtest assumes) so tonight's fit already knows today's error, and re-scored
        as final after local midnight. Missing city-days (bot stopped, other pool cities) are rebuilt from the archives by
        bot/backfill.py, called at startup and daily."""
        if not os.path.exists(C.FORECASTS): return
        F = pd.read_parquet(C.FORECASTS); H = self.history()
        if len(H) and "provisional" not in H.columns: H["provisional"] = False
        if len(H): H["provisional"] = H.provisional.fillna(False).astype(bool)
        done = {(c, str(pd.Timestamp(d).date())): bool(p) for c, d, p in zip(H.city, H.mday, H.provisional)} if len(H) else {}
        add = []; upd = []
        for r in F.itertuples():
            tz = ZoneInfo(TZ[r.city]); day = pd.Timestamp(r.mday).date(); now = dt.datetime.now(tz); key = (r.city, str(day))
            final = now.date() > day; provisional_ok = (now.date() == day and now.hour >= 21)
            if (now.date() - day).days > 3: continue                        # too old for the 72 h METAR feed -> left to backfill
            if key in done and not done[key]: continue                    # already final
            if key in done and done[key] and not final: continue            # provisional and the day is not over yet
            if key not in done and not (final or provisional_ok): continue
            mx, mn = metar_day_max_min(r.city, day, tz, hours=72)
            if pd.isna(mx): continue
            if key in done: upd.append((r.city, day, mx, mx - r.hrrr)); log.info("re-scored %s %s as final: actual %d err %+.1f", r.city, day, mx, mx - r.hrrr)
            else:
                rec = r._asdict(); rec.pop("Index", None); rec.update(actual=mx, err=mx - r.hrrr, provisional=not final); add.append(rec)
                log.info("scored %s %s%s: HRRR %.1f actual %d err %+.1f", r.city, day, "" if final else " (provisional, 21:00 local)", r.hrrr, mx, mx - r.hrrr)
        if add or upd:
            if add: H = pd.concat([H, pd.DataFrame(add)], ignore_index=True)
            H["mday"] = pd.to_datetime(H.mday)
            for c, day, mx, err in upd:
                m = (H.city == c) & (H.mday.dt.date == day); H.loc[m, ["actual", "err", "provisional"]] = [mx, err, False]
            H.to_parquet(C.HISTORY)
    def backfill_history(self):
        """Rebuild any missing city-days of the history from the archives (see bot/backfill.py)."""
        try:
            from bot import backfill
            n = backfill.backfill(); log.info("backfill: %s", f"{n} rows added" if n else "history complete")
        except Exception as ex: log.exception("backfill failed: %s", ex)
        self._last_backfill = dt.datetime.now(dt.timezone.utc).date()

    # ---- market data ----
    def markets_for(self, city, day):
        title = f"Highest temperature in {'NYC' if city=='New York City' else city} on {day.strftime('%B')} {day.day}"
        ev = get_json(f"{C.GAMMA}/events", {"tag_id": 84, "closed": "false", "limit": 100, "title_search": f"Highest temperature in {city}"})
        for e in ev:
            if e["title"].startswith(title) and e["title"].split(" on ")[1].strip("?") == f"{day.strftime('%B')} {day.day}":
                out = {}
                for m in e["markets"]:
                    b = parse_bucket(m["question"]); toks = json.loads(m.get("clobTokenIds") or "[]")
                    if b and len(toks) == 2: out[b] = dict(yes_token=toks[0], no_token=toks[1], cid=m["conditionId"], question=m["question"])
                return out
        return {}
    def best_ask(self, token):
        try:
            ob = self.client.get_order_book(token); asks = sorted(((float(a.price), float(a.size)) for a in ob.asks), key=lambda x: x[0]); return asks
        except Exception as ex: log.warning("book %s: %s", token[:10], ex); return []

    # ---- one city evening ----
    def evaluate(self, city):
        import random
        tz = ZoneInfo(TZ[city]); now = dt.datetime.now(tz); wp = window_pos(now)
        if wp is None: return
        minutes_in, target = wp; key = f"{city}|{target}"
        if key not in self.state["done"]:
            skip = random.random() < C.SKIP_PROB; start = random.randint(*C.START_JITTER_MIN); wallet = self.pick_wallet(city, target)
            self.state["done"][key] = {"runs": [], "usd": 0.0, "skip": skip, "start_min": start, "wallet": wallet, "queue": []}
            log.info("%s %s: plan -> %s, start +%d min, wallet #%d (mode %s)", city, target, "SKIP (camouflage)" if skip else "trade", start, wallet, C.WALLET_MODE); self.save()
        st = self.state["done"][key]
        if st["skip"]: return
        if minutes_in < st["start_min"]: return
        if target.strftime("%Y%m%d") + "00" in st["runs"]: return           # this target's 00Z run already evaluated
        H = self.history()
        if len(H) < C.MIN_HISTORY_DAYS: log.warning("history too short (%d)", len(H)); return
        hrrr, run = hrrr_afternoon_max(city, target, tz)
        if hrrr is None: log.info("%s: 00Z HRRR for %s not complete yet, will retry", city, target); return
        run_id = run.strftime("%Y%m%d%H")
        log.info("%s: using HRRR run %s (00Z of target date, as in history)", city, run.strftime("%Y-%m-%d %HZ"))
        if run_id in st["runs"]: return                                   # already evaluated on this run
        f = openmeteo_fields(city, target, tz); today_max, today_min = metar_day_max_min(city, now.date(), tz, hours=30)
        Hc = H[H.city == city].sort_values("mday")
        # yesterday's error = today's actual - the HRRR we forecast for today (if we have it), else last known err
        last_err = float(Hc.err.iloc[-1]) if len(Hc) else 0.0
        feats = dict(hrrr=hrrr, yday_max=today_max, yday_min=today_min, yday_err=last_err, e2=float(Hc.err.iloc[-2]) if len(Hc) > 1 else last_err,
                     r5=float(Hc.err.tail(5).mean()) if len(Hc) else 0.0, r14=float(Hc.err.tail(14).mean()) if len(Hc) else 0.0, doy=target.timetuple().tm_yday, **f)
        feats["dpd"] = hrrr - feats["dew"]
        nbm = mos_max(city, "NBS", target, tz); gfs = mos_max(city, "GFS", target, tz); feats["nbm_minus_hrrr"] = nbm - hrrr if pd.notna(nbm) else np.nan; feats["gfs_minus_hrrr"] = gfs - hrrr if pd.notna(gfs) else np.nan
        M = Models(H); p = M.predict(city, feats)
        mu_e, mu_r = hrrr + p["ewma"], hrrr + p["ridge"]
        mk = self.markets_for(city, target)
        if not mk: log.warning("%s %s: no markets found", city, target); return
        buckets = list(mk); Pe = bucket_probs(mu_e, p["ewma_sd"], buckets); Pr = bucket_probs(mu_r, p["ridge_sd"], buckets)
        be, br = max(Pe, key=Pe.get), max(Pr, key=Pr.get); agree = be == br
        log.info("%s %s | HRRR run %sZ max %.1f | EWMA %+.1f -> %.1f (sd %.1f) best %s | ridge %+.1f -> %.1f (sd %.1f) best %s | %s", city, target, run.strftime("%d %H"), hrrr, p["ewma"], mu_e, p["ewma_sd"], be, p["ridge"], mu_r, p["ridge_sd"], br, "AGREE" if agree else "DISAGREE")
        # record the forecast for later scoring
        F = pd.read_parquet(C.FORECASTS) if os.path.exists(C.FORECASTS) else pd.DataFrame()
        F = pd.concat([F[~((F.city == city) & (F.mday.astype(str) == str(target)))] if len(F) else F, pd.DataFrame([dict(city=city, mday=pd.Timestamp(target), run=run_id, **feats)])], ignore_index=True); F.to_parquet(C.FORECASTS)
        # decide
        orders = []; modes = C.MODES.get(city, {"agree", "edge"}); yes_ask = {}
        for b in buckets:
            asks = self.best_ask(mk[b]["yes_token"]); 
            if not asks: continue
            ask = asks[0][0]; pav = (Pe[b] + Pr[b]) / 2; yes_ask[b] = ask
            if agree and "agree" in modes and b == be and C.AGREE_MIN_PRICE <= ask <= C.AGREE_MAX_PRICE: orders.append((b, ask, asks, "agree", Pe[b], Pr[b], mk[b]["yes_token"]))
            elif not agree and "edge" in modes and pav - ask >= C.DISAGREE_EDGE and ask <= C.DISAGREE_MAX_PRICE: orders.append((b, ask, asks, "disagree", Pe[b], Pr[b], mk[b]["yes_token"]))
            elif not agree and "ridge" in modes and b == br and C.MODEL_MIN_PRICE <= ask <= C.MODEL_MAX_PRICE: orders.append((b, ask, asks, "dis_ridge", Pe[b], Pr[b], mk[b]["yes_token"]))
            elif not agree and "ewma" in modes and b == be and C.MODEL_MIN_PRICE <= ask <= C.MODEL_MAX_PRICE: orders.append((b, ask, asks, "dis_ewma", Pe[b], Pr[b], mk[b]["yes_token"]))
            time.sleep(0.05)
        if not orders: log.info("%s: no bucket meets the rule (modes %s; best asks: %s)", city, sorted(modes), {b: round(yes_ask[b], 3) if b in yes_ask else None for b in [be, br]})
        # NO leg (rule H): only alongside a disagree-model YES leg. A: NO on the favorite if warmer than our YES bucket;
        # B: ridge cities, NO on the bucket 1 warmer than the ridge pick. That bucket's YES ask must be in [NO_LEG_MIN_YES, NO_LEG_MAX_YES].
        dis_legs = [o for o in orders if o[3] in ("dis_ridge", "dis_ewma")]
        if C.NO_LEG and dis_legs and yes_ask:
            yb = dis_legs[0][0]; cands = {}
            fav = max(yes_ask, key=yes_ask.get)
            if fav[0] > yb[0]: cands[fav] = "no_fav"
            if any(o[3] == "dis_ridge" for o in dis_legs):
                nb = (br[0] + 2, br[1] + 2) if br[1] < 999 else None
                if nb in yes_ask and nb not in cands: cands[nb] = "no_warm"
            for b, why in cands.items():
                if not (C.NO_LEG_MIN_YES <= yes_ask[b] <= C.NO_LEG_MAX_YES): log.info("%s: NO leg %s skipped, %s YES ask %.2f outside %.2f-%.2f", city, why, b, yes_ask[b], C.NO_LEG_MIN_YES, C.NO_LEG_MAX_YES); continue
                nasks = self.best_ask(mk[b]["no_token"])
                if nasks and nasks[0][0] <= 1 - C.NO_LEG_MIN_YES + 0.03: orders.append((b, nasks[0][0], nasks, why, Pe[b], Pr[b], mk[b]["no_token"]))
                else: log.info("%s: NO leg %s skipped, %s NO ask %s", city, why, b, nasks[0][0] if nasks else None)
            if not cands: log.info("%s: NO leg not applicable (favorite %s at %.2f not warmer than our %s)", city, fav, yes_ask[fav], yb)
        # decoy: occasionally a small buy on the 2nd-most-likely bucket (noise in the fill history)
        if random.random() < C.DECOY_PROB:
            ranked = sorted(buckets, key=lambda b: -(Pe[b] + Pr[b])); second = ranked[1] if len(ranked) > 1 else None
            if second and second not in [o[0] for o in orders]:
                asks = self.best_ask(mk[second]["yes_token"])
                if asks and asks[0][0] <= 0.5: orders.append((second, asks[0][0], asks, "decoy", Pe[second], Pr[second], mk[second]["yes_token"]))
        t0 = time.time(); yes_stake = None
        for b, ask, asks, why, pe, pr, token in orders:
            edge = (pe + pr) / 2 - ask
            stake = random.uniform(*C.DECOY_STAKE) if why == "decoy" else C.STAKE * (2 if (why == "disagree" and edge >= 2 * C.DISAGREE_EDGE) else 1) * random.uniform(*C.SIZE_JITTER)
            if why.startswith("no_") and C.NO_LEG_SHARE_MATCH and yes_stake: stake = yes_stake[0] / yes_stake[1] * ask   # same share count as the YES leg
            elif not why.startswith("no_") and why != "decoy": yes_stake = (stake, ask)
            room = min(C.MAX_PER_MARKET_USD - self.state["positions"].get(mk[b]["cid"], {}).get("usd", 0.0), C.MAX_PER_CITY_DAY_USD - st["usd"], C.MAX_DAILY_USD - self.daily_usd())
            stake = min(stake, room)
            if os.path.exists(C.KILL_FILE) or stake < C.MIN_ORDER_SHARES * ask: log.info("skip %s (%s): room $%.0f / kill", mk[b]["question"][40:80], why, room); continue
            n_child = random.randint(*C.CHILD_ORDERS); when = t0 + random.uniform(0, 3) * 60
            for i in range(n_child):
                st["queue"].append(dict(cid=mk[b]["cid"], question=mk[b]["question"], token=token, bucket=list(b), why=why, pe=pe, pr=pr, city=city, target=str(target),
                                        wallet=(self.pick_wallet(city, target, why) if C.WALLET_MODE == "per_signal" else st["wallet"]),
                                        usd=stake / n_child, place_at=when, rest_min=random.uniform(*C.REST_MIN), status="queued", order_id=None, price=None, shares=0.0))
                when += random.uniform(*C.CHILD_GAP_MIN) * 60
            log.info("PLAN %s | %s | P_ewma %.2f P_ridge %.2f ask %.3f | $%.0f in %d child order(s), first at +%.0f min", mk[b]["question"][40:], why, pe, pr, ask, stake, n_child, (t0 + 0 - time.time()) / 60 + (st["queue"][-n_child]["place_at"] - t0) / 60)
        st["runs"].append(run_id); self.save()

    # ---- execution: resting limit 1 tick under the ask, then cross ----------------------------------------
    def expire_stale(self):
        """After downtime: queued orders whose evening window is over are dropped; live resting orders that are still open on the
        exchange are cancelled (fills that happened while offline are recorded first). Runs at startup and every loop."""
        for key, st in self.state["done"].items():
            for o in st.get("queue", []):
                if o["status"] not in ("queued", "resting"): continue
                tz = ZoneInfo(TZ[o["city"]]); target = dt.date.fromisoformat(o["target"]); now_l = dt.datetime.now(tz)
                window_end = dt.datetime(target.year, target.month, target.day, 0, tzinfo=tz) - dt.timedelta(days=1) + dt.timedelta(hours=C.WINDOW[1])   # end of the window that targets this day
                if now_l < window_end + dt.timedelta(minutes=30): continue
                if o["status"] == "resting" and not C.DRY_RUN and o.get("order_id"):
                    try:
                        cl = self.wallet_client(o["wallet"]); got = float(cl.get_order(o["order_id"]).get("size_matched") or 0)
                        if got > 0: self._fill(st, o, o["price"], got, final=False); log.info("offline fill recorded: %.2f sh %s", got, o["question"][40:80])
                        cl.cancel(o["order_id"])
                    except Exception as ex: log.warning("expire/cancel %s: %s", o["question"][40:80], ex)
                o["status"] = "expired"; log.info("expired stale %s order: %s (%s)", o["why"], o["question"][40:80], key)
        self.save()

    def manage_orders(self):
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        now = time.time(); self.expire_stale()
        for key, st in self.state["done"].items():
            for o in st.get("queue", []):
                if o["status"] in ("filled", "cancelled", "abandoned", "expired"): continue
                asks = self.best_ask(o["token"])
                if not asks: continue
                ask, depth = asks[0][0], sum(sz for p_, sz in asks if p_ <= asks[0][0] + 0.01)
                if o["status"] == "queued" and now >= o["place_at"]:
                    if not self.ensure_cash(o["wallet"], o["usd"]): log.warning("skip %s: wallet #%d short of cash even after redeeming", o["question"][40:80], o["wallet"]); o["status"] = "abandoned"; continue
                    shares = round(o["usd"] / ask, 2)                                          # resting limit: full child size, no depth cap
                    if shares < C.MIN_ORDER_SHARES: o["status"] = "abandoned"; log.info("abandon %s: below min order", o["question"][40:80]); continue
                    o["price"] = round(ask - 0.01, 3); o["shares"] = shares; o["rest_until"] = now + o["rest_min"] * 60
                    if not C.DRY_RUN:
                        try:
                            cl = self.wallet_client(o["wallet"]); opts = PartialCreateOrderOptions(tick_size=cl.get_tick_size(o["token"]), neg_risk=cl.get_neg_risk(o["token"]))
                            r = cl.post_order(cl.create_order(OrderArgs(token_id=o["token"], price=o["price"], size=shares, side="BUY"), opts), OrderType.GTC); o["order_id"] = r.get("orderID") or r.get("id")
                        except Exception as ex: log.error("rest order failed: %s", ex); o["status"] = "abandoned"; continue
                    o["status"] = "resting"; log.info("REST bid %.3f x %.2f sh %s (%s) for %.0f min [wallet #%d, %s]", o["price"], shares, o["question"][40:80], o["why"], o["rest_min"], o["wallet"], "DRY" if C.DRY_RUN else "LIVE")
                elif o["status"] == "resting":
                    filled = 0.0
                    if C.DRY_RUN: filled = o["shares"] if ask <= o["price"] else 0.0          # paper: filled if the ask came down to our bid
                    else:
                        try: filled = float(self.wallet_client(o["wallet"]).get_order(o["order_id"]).get("size_matched") or 0)
                        except Exception: pass
                    if filled >= o["shares"] * 0.999: self._fill(st, o, o["price"], o["shares"]); continue
                    if now >= o["rest_until"]:
                        if not C.DRY_RUN:
                            try: self.wallet_client(o["wallet"]).cancel(o["order_id"])
                            except Exception as ex: log.warning("cancel: %s", ex)
                        if filled > 0: self._fill(st, o, o["price"], filled, final=False)
                        remaining = round(min(o["shares"] - filled, C.DEPTH_CAP * depth), 2)          # cross leg: depth cap per child order
                        if remaining < C.MIN_ORDER_SHARES: o["status"] = "filled" if filled else "cancelled"; log.info("no cross %s: %.1f sh unfilled, only %.1f sh at <= %.3f", o["question"][40:80], o["shares"] - filled, depth, ask + 0.01); continue
                        if remaining < o["shares"] - filled - 0.01: log.info("cross capped %s: %.2f of %.2f sh (%.0f%% of %.1f sh depth)", o["question"][40:80], remaining, o["shares"] - filled, C.DEPTH_CAP * 100, depth)
                        got = remaining
                        if not C.DRY_RUN:
                            try:
                                cl = self.wallet_client(o["wallet"]); opts = PartialCreateOrderOptions(tick_size=cl.get_tick_size(o["token"]), neg_risk=cl.get_neg_risk(o["token"]))
                                r = cl.post_order(cl.create_order(OrderArgs(token_id=o["token"], price=round(ask + 0.01, 3), size=remaining, side="BUY"), opts), OrderType.FAK); time.sleep(0.5)
                                got = float(cl.get_order(r.get("orderID") or r.get("id")).get("size_matched") or 0)
                            except Exception as ex: log.error("cross failed: %s", ex); got = 0.0
                        log.info("CROSS at %.3f x %.2f sh %s [%s]", ask, remaining, o["question"][40:80], "DRY" if C.DRY_RUN else "LIVE")
                        if got > 0: self._fill(st, o, ask, got)
                        else: o["status"] = "cancelled"
            self.save()

    def _fill(self, st, o, price, shares, final=True):
        usd = price * shares; p = self.state["positions"].setdefault(o["cid"], {"question": o["question"], "shares": 0.0, "usd": 0.0}); p["shares"] += shares; p["usd"] += usd
        st["usd"] += usd; self.state["daily"][dt.date.today().isoformat()] = self.daily_usd() + usd
        if final: o["status"] = "filled"
        new = not os.path.exists(C.TRADE_LOG)
        with open(C.TRADE_LOG, "a", newline="") as fh:
            w = csv.writer(fh)
            if new: w.writerow(["time_utc", "city", "target_day", "question", "why", "p_ewma", "p_ridge", "ask", "shares", "usd", "dry_run", "wallet"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(), o["city"], o["target"], o["question"], o["why"], round(o["pe"], 3), round(o["pr"], 3), price, round(shares, 2), round(usd, 2), C.DRY_RUN, o["wallet"]])
        log.info("FILL %s | %s | %.2f sh @ %.3f = $%.0f [wallet #%d, %s]", o["question"][40:], o["why"], shares, price, usd, o["wallet"], "DRY" if C.DRY_RUN else "LIVE")

    def run(self):
        self.expire_stale(); self.backfill_history()
        while True:
            try:
                now = dt.datetime.now(dt.timezone.utc)
                if now.hour >= 9 and getattr(self, "_last_backfill", None) != now.date(): self.backfill_history()   # daily, once every US city is past local midnight
                self.score_pending()
                for city in C.CITIES:
                    if window_pos(dt.datetime.now(ZoneInfo(TZ[city]))) is not None: self.evaluate(city)
                self.manage_orders()
            except Exception as ex: log.exception("loop error: %s", ex)
            time.sleep(C.LOOP_SECONDS)

if __name__ == "__main__":
    Bot().run()
