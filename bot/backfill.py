"""Rebuild missing city-days of the bot's history (data/evening_history.parquet) from the archives, so a stopped bot does
not leave holes in the EWMA/ridge training data. Same row construction as backtest_evening.py's panel:
  HRRR 00Z afternoon max (AWS via herbie), METAR daily max/min (IEM ASOS), Open-Meteo previous-runs surface fields
  (previous_day1), IEM NBM/GFS-MOS maxes; lag features recomputed per city on the merged series.
Usage: python3 bot/backfill.py [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--dry]   (also called by the bot at startup / daily)"""
import os, sys, re, json, io, time, subprocess, datetime as dt, requests, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep_local import TZ
from bot import evening_config as C
rh = lambda x: int(Decimal(str(x)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = ['Atlanta','Austin','Houston','Los Angeles','Miami','Seattle','Chicago','Dallas','New York City','Denver','San Francisco']   # the history's 11 cities (ridge pool)
OLD7 = ['Houston','Los Angeles','Atlanta','Miami','Austin','Seattle','San Francisco']; NEW4 = ['New York City','Chicago','Dallas','Denver']
site = json.load(open(f'{ROOT}/out/stations.json')); coords = json.load(open(f'{ROOT}/out/station_coords.json'))
S = requests.Session(); S.headers['User-Agent'] = 'Mozilla/5.0'
def log(*a): print(dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S'), 'backfill:', *a, flush=True)

def refresh_archives(d0, d1):
    """Pull HRRR 00Z, METAR, Open-Meteo previous-runs and MOS for d0..d1 into the data/ archives (all resumable)."""
    env = dict(os.environ, HRRR_START=str(d0), HRRR_END=str(d1), HRRR_00Z_ONLY='1')
    for scr in ('fetch_hrrr_back.py', 'fetch_hrrr_new4.py'):
        try: subprocess.run([sys.executable, scr], cwd=ROOT, env=env, check=False, capture_output=True, timeout=1800)
        except subprocess.TimeoutExpired: log('HRRR fetch timed out:', scr, '(partial data kept; will resume next run)')
    # METAR (IEM ASOS, hourly + specials)
    m = pd.read_parquet(f'{ROOT}/data/metar.parquet'); rows = []
    for c in POOL:
        icao, unit = site[c]; stid = icao[1:] if icao.startswith('K') else icao; f = d0 - dt.timedelta(days=1); t = d1 + dt.timedelta(days=1)
        r = ''
        for att in range(4):
            try:
                rr = S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params={"station": stid, "data": "metar", "year1": f.year, "month1": f.month, "day1": f.day, "year2": t.year, "month2": t.month, "day2": t.day, "tz": TZ[c], "format": "onlycomma", "report_type": "3,4"}, timeout=300)
                if rr.status_code == 200 and len(rr.text) > 200: r = rr.text; break
            except Exception as ex: pass
            time.sleep(5 * (att + 1))
        if not r: log('METAR fail', c); continue
        for line in r.splitlines():
            if line.startswith("station,") or not line.strip(): continue
            p = line.split(",", 2)
            if len(p) < 3: continue
            raw = p[2]; body = re.search(r"\s(M?\d\d)/(M?\d\d)?\s", raw); tg = re.search(r"\sT(\d)(\d{3})(\d)(\d{3})", raw)
            if not body: continue
            cc = (-1 if tg.group(1) == "1" else 1) * int(tg.group(2)) / 10 if tg else float(body.group(1).replace("M", "-"))
            rows.append(dict(city=c, station=icao, unit=unit, local_time=pd.Timestamp(p[1]), temp_c=cc, has_tenths=bool(tg)))
        time.sleep(0.3)
    if rows:
        d = pd.DataFrame(rows); m = pd.concat([m, d]).drop_duplicates(['station', 'local_time'], keep='last').sort_values(['station', 'local_time']); m.to_parquet(f'{ROOT}/data/metar.parquet')
    # Open-Meteo previous-runs surface fields (previous_day1 = the run from the day before)
    V = ['temperature_2m','dew_point_2m','relative_humidity_2m','wind_speed_10m','wind_direction_10m','cloud_cover','shortwave_radiation','surface_pressure','precipitation']
    w = pd.read_parquet(f'{ROOT}/data/hrrr_wx_d1.parquet'); new = []
    for c in POOL:
        lat, lon = coords[site[c][0]]
        try: j = S.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "hourly": ",".join(v + "_previous_day1" for v in V), "start_date": str(d0), "end_date": str(d1), "models": "gfs_hrrr", "timezone": TZ[c], "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}, timeout=90).json()
        except Exception as ex: log('Open-Meteo fail', c, ex); continue
        if 'hourly' not in j: continue
        h = pd.DataFrame({'time': pd.to_datetime(j['hourly']['time']), **{v: j['hourly'][v + '_previous_day1'] for v in V}}); h['city'] = c; new.append(h); time.sleep(0.3)
    if new:
        new = pd.concat(new).dropna(subset=['temperature_2m']); w = pd.concat([w[~(w.city.isin(POOL) & (w.time >= pd.Timestamp(d0)) & (w.time < pd.Timestamp(d1) + pd.Timedelta(days=1)))], new]).sort_values(['city', 'time']); w.to_parquet(f'{ROOT}/data/hrrr_wx_d1.parquet')
    # IEM MOS
    mo = pd.read_parquet(f'{ROOT}/data/mos_nbm.parquet'); rows = []
    for c in POOL:
        for model in ['NBS', 'GFS']:
            try: r = S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py", params={"station": site[c][0], "model": model, "sts": f"{d0 - dt.timedelta(days=1)}T00:00Z", "ets": f"{d1}T23:00Z", "format": "csv"}, timeout=120)
            except Exception as ex: log('MOS fail', c, model, ex); continue
            if r.status_code == 200 and len(r.text) > 100:
                d = pd.read_csv(io.StringIO(r.text)); d['city'] = c; rows.append(d)
            time.sleep(0.3)
    if rows:
        d = pd.concat(rows); d['runtime'] = pd.to_datetime(d.runtime, utc=True).dt.tz_localize(None).astype(str); d['ftime'] = pd.to_datetime(d.ftime, utc=True).dt.tz_localize(None).astype(str)
        d = d[[k for k in mo.columns if k in d.columns]]; mo = pd.concat([mo, d]).drop_duplicates(['runtime', 'ftime', 'model', 'station'], keep='last'); mo.to_parquet(f'{ROOT}/data/mos_nbm.parquet')

def build_rows(days_needed):
    """days_needed: dict city -> set of dates. Returns rows in the history schema (lag features filled later)."""
    h = pd.concat([pd.read_parquet(f'{ROOT}/data/{f}') for f in ('hrrr_t2m.parquet', 'hrrr_t2m_east.parquet') if os.path.exists(f'{ROOT}/data/{f}')])
    h = h[(h.city != 'ERR') & (h.run == 0)].drop_duplicates(['date', 'run', 'fxx', 'city']).copy(); h['valid_utc'] = pd.to_datetime(h.valid_utc, utc=True)
    h['local'] = [t.tz_convert(TZ[ct]) for t, ct in zip(h.valid_utc, h.city)]; h['mday'] = [x.date() for x in h.local]; h['lh'] = [x.hour for x in h.local]
    cnt = h[h.lh.between(11, 18)].groupby(['city', 'mday']).lh.nunique(); hm = h[h.lh.between(11, 18)].groupby(['city', 'mday']).t2m_f.max(); hm = hm[cnt >= 6]
    met = pd.read_parquet(f'{ROOT}/data/metar.parquet'); met['day'] = met.local_time.dt.date; met['tf'] = (met.temp_c * 9 / 5 + 32).map(rh)
    dmax = met.groupby(['station', 'day']).tf.max(); dmin = met.groupby(['station', 'day']).tf.min()
    w = pd.read_parquet(f'{ROOT}/data/hrrr_wx_d1.parquet'); w['mday'] = w.time.dt.date; w['lh'] = w.time.dt.hour; aft = w[w.lh.between(11, 18)]
    wx = aft.groupby(['city', 'mday']).agg(dew=('dew_point_2m', 'mean'), rhum=('relative_humidity_2m', 'mean'), wind=('wind_speed_10m', 'mean'), cloud=('cloud_cover', 'mean'), rad=('shortwave_radiation', 'mean'), pres=('surface_pressure', 'mean'), precip=('precipitation', 'sum'),
                                          wdir=('wind_direction_10m', lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(), np.cos(np.radians(s)).mean())) % 360))
    morn = w[w.lh.between(4, 8)].groupby(['city', 'mday']).agg(t_morn=('temperature_2m', 'mean'), cloud_morn=('cloud_cover', 'mean'))
    mos = pd.read_parquet(f'{ROOT}/data/mos_nbm.parquet'); mos['runtime'] = pd.to_datetime(mos.runtime, utc=True); mos['ftime'] = pd.to_datetime(mos.ftime, utc=True)
    def mos_daily(model):
        d = mos[(mos.model == model) & (mos.runtime.dt.hour.isin([0, 1]))].copy(); out = {}
        for c, g in d.groupby('city'):
            tz = ZoneInfo(TZ[c]); g = g.copy(); lt = g.ftime.dt.tz_convert(tz); g['md'] = lt.dt.date; g['lh'] = lt.dt.hour; g = g[g.lh.between(11, 18)]
            for md, gg in g.groupby('md'):
                if md not in days_needed.get(c, ()): continue
                cutoff = pd.Timestamp(dt.datetime(md.year, md.month, md.day, 21, tzinfo=tz) - dt.timedelta(days=1)); gg = gg[gg.runtime <= cutoff]
                if gg.empty: continue
                last = gg[gg.runtime == gg.runtime.max()]; out[(c, md)] = pd.to_numeric(last.tmp, errors='coerce').max()
        return out
    nbs = mos_daily('NBS'); gfs = mos_daily('GFS'); rows = []
    for c, days in days_needed.items():
        st = site[c][0]
        for d in sorted(days):
            v = hm.get((c, d)); act = dmax.get((st, d))
            if v is None or act is None or pd.isna(act): continue
            r = dict(city=c, mday=pd.Timestamp(d), hrrr=float(v), actual=int(act), err=float(act - v), yday_max=dmax.get((st, d - dt.timedelta(days=1)), np.nan), yday_min=dmin.get((st, d - dt.timedelta(days=1)), np.nan), doy=d.timetuple().tm_yday,
                     nbm_max=nbs.get((c, d), np.nan), gfsmos_max=gfs.get((c, d), np.nan), run=f"{d:%Y%m%d}00")
            for src in (wx, morn):
                if (c, d) in src.index: r.update(src.loc[(c, d)].to_dict())
            rows.append(r)
    return pd.DataFrame(rows)

def add_lag_features(H):
    H = H.sort_values(['city', 'mday']).reset_index(drop=True)
    H['yday_err'] = H.groupby('city').err.shift(1); H['e2'] = H.groupby('city').err.shift(2)
    H['r5'] = H.groupby('city').err.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean()); H['r14'] = H.groupby('city').err.transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
    H['dpd'] = H.hrrr - H.dew; H['wdir_s'] = np.sin(np.radians(H.wdir)); H['wdir_c'] = np.cos(np.radians(H.wdir)); H['nbm_minus_hrrr'] = H.nbm_max - H.hrrr; H['gfs_minus_hrrr'] = H.gfsmos_max - H.hrrr
    return H

def backfill(d_from=None, d_to=None, dry=False):
    H = pd.read_parquet(C.HISTORY if os.path.isabs(C.HISTORY) else f'{ROOT}/{C.HISTORY}'); H['mday'] = pd.to_datetime(H.mday)
    have = set(zip(H.city, H.mday.dt.date)); yesterday = min(dt.datetime.now(ZoneInfo(TZ[c])).date() for c in POOL) - dt.timedelta(days=1)
    d1 = d_to or yesterday
    last_per_city = H[H.city.isin(POOL)].groupby('city').mday.max().dt.date
    d0 = d_from or max(min(last_per_city) - dt.timedelta(days=1), d1 - dt.timedelta(days=120))       # from the oldest city's last row (cap 120 days)
    if (d1 - d0).days > 30: d1 = d0 + dt.timedelta(days=30); log(f'large gap: filling {d0}..{d1} this run, the rest on the next runs')
    need = {c: {d for d in pd.date_range(d0, d1).date if (c, d) not in have} for c in POOL}; need = {c: v for c, v in need.items() if v}
    n = sum(len(v) for v in need.values())
    if n == 0: log('history complete through', d1); return 0
    log(f'{n} missing city-days between {d0} and {d1}:', {c: len(v) for c, v in need.items()})
    if dry: return n
    refresh_archives(min(min(v) for v in need.values()), d1)
    new = build_rows(need)
    if new.empty: log('no rows could be built (archives missing?)'); return 0
    new['provisional'] = False
    if 'provisional' not in H.columns: H['provisional'] = False
    keep = [c for c in H.columns if c in new.columns]; H2 = add_lag_features(pd.concat([H, new[keep]], ignore_index=True))
    # only the NEW rows get their lag features from the recomputation; existing rows keep what the bot recorded
    key = set(zip(new.city, new.mday)); mask = [(c, d) in key for c, d in zip(H2.city, H2.mday)]
    out = pd.concat([H[[c for c in H2.columns if c in H.columns]], H2[mask]], ignore_index=True).sort_values(['city', 'mday']).drop_duplicates(['city', 'mday'], keep='first')
    out.to_parquet(C.HISTORY if os.path.isabs(C.HISTORY) else f'{ROOT}/{C.HISTORY}'); log(f'added {int(sum(mask))} rows; history now {len(out)} rows through {out.mday.max().date()}')
    return int(sum(mask))

if __name__ == '__main__':
    a = sys.argv[1:]; kw = {}
    if '--from' in a: kw['d_from'] = dt.date.fromisoformat(a[a.index('--from') + 1])
    if '--to' in a: kw['d_to'] = dt.date.fromisoformat(a[a.index('--to') + 1])
    backfill(dry='--dry' in a, **kw)
