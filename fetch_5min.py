"""5-minute ASOS observations (IEM report_type 1,2) for the US (°F) stations, Jan-Sep 2026."""
import json, os, re, time, requests, pandas as pd
from sweep_local import TZ
site = json.load(open("out/stations.json")); os.makedirs("data/metar5_raw", exist_ok=True)
rows = []
for city, (icao, unit) in site.items():
    if unit != "F": continue
    cache = f"data/metar5_raw/{icao}.csv"
    if not os.path.exists(cache):
        txt = ""
        for m1, m2 in [(1, 3), (3, 5), (5, 7), (7, 9), (9, 10)]:
            for attempt in range(5):
                try:
                    txt += requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                        params={"station": icao[1:], "data": "metar", "year1": 2026, "month1": m1, "day1": 1, "year2": 2026, "month2": m2, "day2": 1,
                                "tz": TZ[city], "format": "onlycomma", "report_type": "1,2"}, timeout=600).text + "\n"; break
                except requests.RequestException as e: print("retry", city, m1, e.__class__.__name__, flush=True); time.sleep(5)
        open(cache, "w").write(txt)
    n = 0
    for line in open(cache):
        if line.startswith("station,") or not line.strip(): continue
        p = line.split(",", 2)
        if len(p) < 3: continue
        raw = p[2]; body = re.search(r"\s(M?\d\d)/(M?\d\d)?\s", raw); tg = re.search(r"\sT(\d)(\d{3})(\d)(\d{3})", raw)
        if not body: continue
        c = (-1 if tg.group(1) == "1" else 1) * int(tg.group(2)) / 10 if tg else float(body.group(1).replace("M", "-"))
        rows.append(dict(city=city, station=icao, local_time=p[1], temp_c=c, has_tenths=bool(tg), tenths_nonzero=bool(tg) and int(tg.group(2)) % 10 != 0)); n += 1
    print(f"{city:<14} {icao} {n} 5-min obs", flush=True)
df = pd.DataFrame(rows); df["local_time"] = pd.to_datetime(df.local_time); df.to_parquet("data/metar5.parquet"); print(len(df))
