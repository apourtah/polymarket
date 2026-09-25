"""Is the day-ahead forecast bias seasonal, and does the pattern repeat across years?

The September 2026 losses trace to a warm shift the models were slow to follow: the ridge's bias ran -0.33
(Feb-Jun) to +0.54 (Sep).  If that autumn swing is a RECURRING seasonal signature rather than one bad year, it is
knowable in advance and there is no reason to make the EWMA rediscover it every September.  This builds four years
of "last night's forecast vs the actual high" for the seven traded cities and asks whether the monthly bias
profile of 2022-2025 says anything about 2026.

Forecast source: Open-Meteo's historical-forecast archive, `temperature_2m_previous_day1` -- the run issued the day
before, i.e. exactly what the bot acts on.  It uses best_match rather than gfs_hrrr, because HRRR's previous_day1
is not archived before ~2025 (probed: 0 non-null hours for 2023, full coverage for best_match back to 2021).  That
makes this a proxy for the bot's own HRRR error, so the script validates it against our real `err` over the months
where both exist before drawing any conclusion from it.

Actuals: IEM ASOS METAR, same stations and the same whole-degree-F rounding the bot uses.

  python3 seasonality.py            # uses the cached fetches when present
  REFETCH=1 python3 seasonality.py  # re-download
"""
import os, sys, re, json, time, datetime as dt
import numpy as np, pandas as pd, requests
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ

rh = lambda x: int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
CITIES = ["Los Angeles", "Austin", "Chicago", "Houston", "Dallas", "Seattle", "Miami"]
SITE = json.load(open("out/stations.json")); COORD = json.load(open("out/station_coords.json"))
START = dt.date(2022, 9, 1); END = dt.date(2026, 9, 24)
FC = "data/seasonality_forecast.parquet"; OB = "data/seasonality_obs.parquet"
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"


def fetch_forecasts():
    """Day-ahead (previous_day1) hourly temperature, per city, from the historical-forecast archive."""
    rows = []
    for c in CITIES:
        lat, lon = COORD[SITE[c][0]]
        for y in range(START.year, END.year + 1):
            a = max(START, dt.date(y, 1, 1)); b = min(END, dt.date(y, 12, 31))
            if a > b: continue
            for attempt in range(5):
                try:
                    j = S.get("https://historical-forecast-api.open-meteo.com/v1/forecast",
                              params=dict(latitude=lat, longitude=lon, hourly="temperature_2m_previous_day1",
                                          start_date=a.isoformat(), end_date=b.isoformat(),
                                          temperature_unit="fahrenheit", timezone=TZ[c]), timeout=120).json()
                    h = j["hourly"]; break
                except Exception as e:
                    print("  retry", c, y, e.__class__.__name__, flush=True); time.sleep(3)
            else:
                continue
            d = pd.DataFrame({"time": pd.to_datetime(h["time"]), "t": h["temperature_2m_previous_day1"]})
            d["city"] = c; rows.append(d)
            print(f"  {c:<13} {y}: {d.t.notna().sum()} hours", flush=True); time.sleep(0.3)
    F = pd.concat(rows).dropna(subset=["t"])
    F.to_parquet(FC); return F


def fetch_obs():
    """Actual daily high from IEM ASOS METAR, whole degrees F, same decoding as the bot."""
    rows = []
    for c in CITIES:
        icao = SITE[c][0]; stid = icao[1:] if icao.startswith("K") else icao
        for y in range(START.year, END.year + 1):
            a = max(START, dt.date(y, 1, 1)); b = min(END + dt.timedelta(days=1), dt.date(y, 12, 31))
            if a >= b: continue
            txt = ""
            for attempt in range(8):
                try:
                    txt = S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                                params={"station": stid, "data": "metar", "year1": a.year, "month1": a.month,
                                        "day1": a.day, "year2": b.year, "month2": b.month, "day2": b.day,
                                        "tz": TZ[c], "format": "onlycomma", "report_type": "3,4"}, timeout=600).text
                    if txt.count("\n") >= 100: break                  # IEM throttles with an EMPTY 200, not an error
                    raise IOError(f"short response ({len(txt)} bytes)")
                except Exception as e:
                    print(f"  retry {c} {y}: {e.__class__.__name__} {e}", flush=True); txt = ""; time.sleep(5 * (attempt + 1))
            if not txt:
                print(f"  GAVE UP {c} {y}", flush=True); continue
            n = 0
            for line in txt.splitlines():
                if line.startswith("station,") or not line.strip(): continue
                p = line.split(",", 2)
                if len(p) < 3: continue
                raw = p[2]; body = re.search(r"\s(M?\d\d)/(M?\d\d)?\s", raw); tg = re.search(r"\sT(\d)(\d{3})(\d)(\d{3})", raw)
                if not body: continue
                cc = (-1 if tg.group(1) == "1" else 1) * int(tg.group(2)) / 10 if tg else float(body.group(1).replace("M", "-"))
                rows.append((c, pd.Timestamp(p[1]), cc)); n += 1
            print(f"  {c:<13} {y}: {n} obs", flush=True); time.sleep(4)
    O = pd.DataFrame(rows, columns=["city", "local_time", "temp_c"])
    O.to_parquet(OB); return O


if __name__ == "__main__":
    refetch = os.environ.get("REFETCH") == "1"
    print("forecasts...", flush=True)
    F = fetch_forecasts() if (refetch or not os.path.exists(FC)) else pd.read_parquet(FC)
    print("observations...", flush=True)
    O = fetch_obs() if (refetch or not os.path.exists(OB)) else pd.read_parquet(OB)

    # day-ahead predicted high = max over 11-18 local, the window the bot uses
    F["d"] = F.time.dt.date; F["h"] = F.time.dt.hour
    pred = F[F.h.between(11, 18)].groupby(["city", "d"]).t.max().rename("pred")
    O["d"] = O.local_time.dt.date; O["tf"] = (O.temp_c * 9 / 5 + 32).map(rh); O["h"] = O.local_time.dt.hour
    g = O.groupby(["city", "d"]).agg(actual=("tf", "max"), hours=("h", "nunique"))
    act = g[g.hours >= 20].actual                                  # complete days only
    D = pd.concat([pred, act], axis=1).dropna().reset_index()
    D["err"] = D.actual - D.pred
    D["month"] = pd.to_datetime(D.d).dt.month; D["year"] = pd.to_datetime(D.d).dt.year
    D["season_year"] = D.year                                       # calendar year is fine for month-on-month work
    print(f"\npaired city-days: {len(D)}  {D.d.min()}..{D.d.max()}")

    # ---- does this proxy track the bot's own HRRR error at all? ----
    H = pd.read_parquet("data/evening_history.parquet"); H["d"] = pd.to_datetime(H.mday).dt.date
    J = D.merge(H[["city", "d", "err"]].rename(columns={"err": "err_hrrr"}), on=["city", "d"])
    print(f"\n=== validation against the bot's own HRRR error, {J.d.min()}..{J.d.max()} ({len(J)} city-days) ===")
    print(f"  correlation of the two daily error series: {J.err.corr(J.err_hrrr):+.3f}")
    print(f"  mean bias  proxy {J.err.mean():+.2f}   HRRR {J.err_hrrr.mean():+.2f}")
    mj = J.groupby(J.d.map(lambda x: x.month)).agg(proxy=("err", "mean"), hrrr=("err_hrrr", "mean")).round(2)
    print("  monthly means:"); print(mj.to_string())
    print(f"  correlation of the MONTHLY profiles: {mj.proxy.corr(mj.hrrr):+.3f}")

    # ---- the question: is the monthly bias profile stable across years? ----
    piv = D.pivot_table(index="month", columns="year", values="err", aggfunc="mean").round(2)
    cnt = D.pivot_table(index="month", columns="year", values="err", aggfunc="size")
    print("\n=== mean day-ahead error (actual - forecast, F) by month and year, all 7 cities ===")
    print(piv.to_string())
    print("\n(city-days behind each cell)"); print(cnt.to_string())
    yrs = [y for y in piv.columns if piv[y].notna().sum() >= 6]
    print("\n=== does one year's monthly profile predict another's? (correlation across months) ===")
    C = piv[yrs].corr().round(3); print(C.to_string())
    off = [C.loc[a, b] for i, a in enumerate(yrs) for b in yrs[i+1:]]
    print(f"\nmean off-diagonal correlation: {np.mean(off):+.3f}")
    if 2026 in yrs:
        oth = [y for y in yrs if y != 2026]
        print(f"2026 vs each earlier year: " + ", ".join(f"{y} {C.loc[2026, y]:+.2f}" for y in oth))
        prior = piv[oth].mean(axis=1)
        m = piv[2026].notna() & prior.notna()
        print(f"2026 vs the average of {oth}: {piv[2026][m].corr(prior[m]):+.3f} over {m.sum()} months")

    print("\n=== September specifically (the month that hurt) ===")
    sep = D[D.month == 9]
    print(sep.groupby(["year"]).err.agg(["size", "mean", "std"]).round(2).to_string())
    print("\nSeptember mean error by city and year:")
    print(sep.pivot_table(index="city", columns="year", values="err", aggfunc="mean").round(2).to_string())
    print("\nfor contrast, the same table for June:")
    print(D[D.month == 6].pivot_table(index="city", columns="year", values="err", aggfunc="mean").round(2).to_string())
    D.to_parquet("out/seasonality_pairs.parquet")
