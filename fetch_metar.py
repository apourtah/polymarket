"""Pull hourly/special METARs for every resolution station, 2026-01-01..today,
from the IEM ASOS archive; parse the T-group tenths. -> data/metar.parquet
columns: city, station, unit, local_time (tz-naive local), temp_c (tenths if
available, else body), tempF/round applied downstream.
"""
import json, os, re, time, requests, datetime as dt, sys
os.makedirs('data/metar_raw', exist_ok=True)
import pandas as pd
from sweep_local import TZ

site = json.load(open("out/stations.json"))
rows = []
for city, (icao, unit) in site.items():
    tz = TZ[city]; stid = icao[1:] if icao.startswith("K") else icao
    cache = f"data/metar_raw/{icao}.csv"
    if os.path.exists(cache):
        r = open(cache).read()
    else:
        r = ""
        for m1, m2 in [(1, 4), (4, 7), (7, 10)]:
            for attempt in range(5):
                try:
                    r += requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                        params={"station": stid, "data": "metar", "year1": 2026, "month1": m1, "day1": 1,
                                "year2": 2026, "month2": m2, "day2": 1, "tz": tz, "format": "onlycomma", "report_type": "3,4"},
                        timeout=300).text + "\n"
                    break
                except requests.RequestException as e:
                    print("  retry", city, m1, e.__class__.__name__, flush=True); time.sleep(5)
        open(cache, "w").write(r)
    n = 0
    for line in r.splitlines():
        if line.startswith("station,") or not line.strip(): continue
        p = line.split(",", 2)
        if len(p) < 3: continue
        raw = p[2]; body = re.search(r"\s(M?\d\d)/(M?\d\d)?\s", raw); tg = re.search(r"\sT(\d)(\d{3})(\d)(\d{3})", raw)
        if not body: continue
        c = (-1 if tg.group(1) == "1" else 1) * int(tg.group(2)) / 10 if tg else float(body.group(1).replace("M", "-"))
        rows.append(dict(city=city, station=icao, unit=unit, local_time=p[1], temp_c=c, has_tenths=bool(tg))); n += 1
    print(f"{city:<16} {icao} {n} obs", flush=True)
df = pd.DataFrame(rows); df["local_time"] = pd.to_datetime(df.local_time)
df.to_parquet("data/metar.parquet"); print(len(df), "observations ->", "data/metar.parquet")
