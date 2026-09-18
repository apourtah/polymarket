"""Live weather feed for today's Polymarket 'highest temperature' markets.

For each city with an open market today:
  station    ICAO station the market resolves on (parsed from the market description)
  obs_max    highest METAR reading so far today (local day), from aviationweather.gov
  obs_last   latest reading and its time
  fc_rest    Open-Meteo forecast max for the remaining hours of the local day
  buckets    the market's buckets with current YES midpoints, and which bucket
             obs_max currently sits in

Usage: python3 weather_feed.py [--json]
"""
import re, sys, json, datetime as dt
from zoneinfo import ZoneInfo
import requests
from sweep_local import TZ, CITY_RE

GAMMA = "https://gamma-api.polymarket.com"; CLOB = "https://clob.polymarket.com"
UA = {"User-Agent": "polymarket-weather-research"}
BUCKET_RE = re.compile(r"be (?:between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC])")

def todays_events():
    ev = []; off = 0
    while True:
        page = requests.get(f"{GAMMA}/events", params={"tag_id": 84, "closed": "false", "limit": 100, "offset": off}, timeout=30).json()
        if not page: break
        ev += [e for e in page if e["title"].startswith("Highest temperature in")]; off += 100
    return ev

def parse_bucket(q):
    m = BUCKET_RE.search(q)
    if not m: return None
    if m.group(1): return (int(m.group(1)), int(m.group(2)))
    if m.group(3): return (int(m.group(3)), 999) if m.group(4) == "higher" else (-999, int(m.group(3)))
    return (int(m.group(5)), int(m.group(5)))

def metar_today(icao, tz, unit):
    r = requests.get("https://aviationweather.gov/api/data/metar", params={"ids": icao, "format": "json", "hours": 30}, timeout=30).json()
    today = dt.datetime.now(tz).date(); obs = []
    for o in r:
        if o.get("temp") is None: continue
        t = dt.datetime.fromisoformat(o["reportTime"].replace("Z", "+00:00")).astimezone(tz)
        if t.date() == today:
            temp = o["temp"] * 9 / 5 + 32 if unit == "F" else o["temp"]
            obs.append((t, round(temp, 1)))
    return sorted(obs)

def forecast_rest(lat, lon, tz, unit):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit" if unit == "F" else "celsius", "timezone": tz.key, "forecast_days": 1}, timeout=30).json()
    now = dt.datetime.now(tz)
    rest = [t for h, t in zip(r["hourly"]["time"], r["hourly"]["temperature_2m"]) if dt.datetime.fromisoformat(h).replace(tzinfo=tz) >= now]
    return max(rest) if rest else None

def station_coords(icao):
    r = requests.get("https://aviationweather.gov/api/data/stationinfo", params={"ids": icao, "format": "json"}, timeout=30).json()
    return (r[0]["lat"], r[0]["lon"]) if r else (None, None)

def main():
    out = []
    for e in todays_events():
        city = e["title"][len("Highest temperature in "):].rsplit(" on ", 1)[0]
        tzname = TZ.get(city)
        if not tzname: continue
        tz = ZoneInfo(tzname)
        desc = e["markets"][0].get("description", "")
        site = re.search(r"site=(\w+)", desc)
        u = re.search(r"in degrees (Celsius|Fahrenheit)", desc); unit = "F" if (u and u.group(1) == "Fahrenheit") else "C"
        day = re.search(r"on (\d{1,2} \w{3} '\d\d)", desc)
        if not site: continue
        icao = site.group(1).upper()
        today = dt.datetime.now(tz).date()
        if day and dt.datetime.strptime(day.group(1), "%d %b '%y").date() != today: continue
        obs = metar_today(icao, tz, unit)
        lat, lon = station_coords(icao)
        fc = forecast_rest(lat, lon, tz, unit) if lat else None
        obs_max = max(t for _, t in obs) if obs else None
        buckets = []
        for m in e["markets"]:
            b = parse_bucket(m["question"]); tok = json.loads(m["clobTokenIds"])[0]
            try: mid = float(requests.get(f"{CLOB}/midpoint", params={"token_id": tok}, timeout=15).json()["mid"])
            except Exception: mid = None
            inb = obs_max is not None and b and b[0] - 0.5 <= obs_max < b[1] + 0.5
            buckets.append(dict(bucket=b, question=m["question"].split(" be ", 1)[1].rstrip("?"), yes_mid=mid, contains_obs_max=inb))
        out.append(dict(city=city, station=icao, unit=unit, local_time=dt.datetime.now(tz).strftime("%H:%M"),
                        obs_max=obs_max, obs_last=(obs[-1][0].strftime("%H:%M"), obs[-1][1]) if obs else None,
                        n_obs=len(obs), fc_rest_max=fc, buckets=buckets))
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1, default=str)); return
    for c in out:
        hot = [b for b in c["buckets"] if b["contains_obs_max"]]
        print(f"{c['city']:<16} {c['station']} {c['local_time']} local | obs max {c['obs_max']}°{c['unit']} (last {c['obs_last']}, {c['n_obs']} obs) | "
              f"fc rest-of-day max {c['fc_rest_max']} | obs-max bucket: {hot[0]['question'] if hot else '?'} @ YES {hot[0]['yes_mid'] if hot else '-'}")

if __name__ == "__main__":
    main()
