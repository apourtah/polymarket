"""Night-before strategy: take the day-1-ahead forecast daily max (ECMWF IFS via
Open-Meteo previous-runs), buy NO on the buckets 2 degrees below and 2 degrees
above it (°C cities: exact-degree buckets; °F cities: 2°F buckets -> 2 buckets
away = +-4°F). Loss = that bucket wins. Prices: NO midpoint at 21:00 local the
evening before, from the 5-min price archive."""
import json, re, time, bisect, datetime as dt, requests, pandas as pd, numpy as np
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ, CITY_RE, EXCL
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
site=json.load(open('out/stations.json')); coords=json.load(open('out/station_coords.json'))
MODEL="ecmwf_ifs025"
# ---- forecasts: day-1-ahead hourly -> daily max per station-day
fc={}
for city,(st,unit) in site.items():
    lat,lon=coords[st]; tz=TZ[city]
    for attempt in range(4):
        r=requests.get("https://previous-runs-api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m_previous_day1","start_date":"2026-01-01","end_date":"2026-09-16","models":MODEL,"timezone":tz},timeout=60)
        if r.status_code==200: break
        time.sleep(3)
    h=r.json()['hourly']; s=pd.Series(h['temperature_2m_previous_day1'],index=pd.to_datetime(h['time']))
    dm=s.groupby(s.index.date).max()
    for d,v in dm.items():
        if pd.notna(v): fc[(city,d)]=v
    time.sleep(0.3)
print(len(fc),"station-day forecasts")
# ---- actual hourly max
m1=pd.read_parquet('data/metar.parquet'); m1['day']=m1.local_time.dt.date
act={}
for (st,d),g in m1.groupby(['station','day']): act[(st,d)]=g.temp_c.max()
# ---- markets by (city, day, bucket lo)
ms=[m for m in json.load(open('data/markets.json')) if CITY_RE.match(m['question'] or '') and not EXCL.search(m['question'])]
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    return (int(mm.group(4)),int(mm.group(4)))
mk={}
for m in ms:
    city=CITY_RE.match(m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); b=bucket(m['question'])
    mk[(city,d,b[0])]=(m,b)
def no_price(m, ts):
    try: h=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    if not h['history']: return None
    hs=sorted((x['t'],x['p']) for x in h['history']); i=bisect.bisect_right([t for t,_ in hs],ts)-1
    if i<0: return None
    p=hs[i][1]; return p if h['losing_outcome']=='No' else 1-p
rows=[]
for (city,d),f in fc.items():
    st,unit=site[city]
    if (st,d) not in act: continue
    a=act[(st,d)]; tz=ZoneInfo(TZ[city])
    if unit=='F': F=rh(f*9/5+32); A=rh(a*9/5+32); step=2; F=F-(F%2)      # snap forecast to the 2°F bucket floor (even)
    else: F=rh(f); A=rh(a); step=1
    ts=int(dt.datetime(d.year,d.month,d.day,21,tzinfo=tz).timestamp())-86400   # 21:00 local the evening before
    for side,lo in (('-2',F-2*step),('+2',F+2*step)):
        key=(city,d,lo)
        if key not in mk: continue
        m,b=mk[key]; won=[float(x) for x in m['outcome_prices']]==[1.0,0.0]   # bucket (YES) won -> our NO loses
        rows.append(dict(city=city,unit=unit,day=str(d),side=side,fc_max=F,actual=A,bucket_lo=lo,bucket_hi=b[1],lose=won,no_px=no_price(m,ts),resolved=sorted(float(x) for x in m['outcome_prices'])==[0.0,1.0]))
df=pd.DataFrame(rows); df=df[df.resolved]; df.to_csv('out/forecast_pm2.csv',index=False)
pd.set_option('display.width',200)
for u in ['C','F']:
    s=df[df.unit==u]
    print(f"\n=== {u} cities: {len(s)} bucket-trades over {s.groupby(['city','day']).ngroups} station-days")
    print("P(lose) by side:", s.groupby('side').lose.mean().map('{:.2%}'.format).to_dict(), "| either side loses on a day:", f"{s.groupby(['city','day']).lose.max().mean():.2%}")
    print("forecast error (actual - forecast, rounded):", (s.drop_duplicates(['city','day']).actual-s.drop_duplicates(['city','day']).fc_max).value_counts(normalize=True).sort_index().loc[-5:5].map('{:.1%}'.format).to_dict())
    p=s[s.no_px.notna()].copy(); p['px']=(p.no_px+0.005).clip(upper=0.999); p['sh']=50/p.px; p['fee']=p.sh*0.05*p.px*(1-p.px); p['pnl']=np.where(p.lose,0,p.sh)-50-p.fee
    print(f"NO price at 21:00 the night before: mean {p.no_px.mean():.3f}, p25 {p.no_px.quantile(.25):.3f}, p75 {p.no_px.quantile(.75):.3f} | trades with a price {len(p)}")
    print(f"$50/trade: P&L ${p.pnl.sum():,.0f} on ${50*len(p):,.0f} ({p.pnl.sum()/(50*len(p)):+.2%}), loss rate {p.lose.mean():.2%}, breakeven loss rate {(1-p.px).mean():.2%}")
    print("  by side:", p.groupby('side').apply(lambda g: f"n={len(g)} px={g.no_px.mean():.3f} lose={g.lose.mean():.2%} roi={g.pnl.sum()/(50*len(g)):+.2%}").to_dict())
    print("  by NO price band:"); print(p.groupby(pd.cut(p.no_px,[0,.7,.8,.9,.95,1.0]),observed=True).apply(lambda g: pd.Series(dict(n=len(g),lose=round(g.lose.mean(),4),breakeven=round((1-g.px).mean(),4),roi=round(g.pnl.sum()/(50*len(g)),4)))).to_string())
