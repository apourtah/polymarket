"""Calibrated-forecast YES strategy (their structure, our forecast).
Per station: empirical distribution of (actual hourly max - ECMWF day-ahead max), rounded in market units,
learned on TRAIN days. On TEST days: P(bucket) = sum of that distribution over the bucket's range; buy $50 YES
on any bucket where P >= market YES price at ENTRY_H local + EDGE. Fill = ask ~ mid + 0.5c; fee 0.05 p(1-p)."""
import json, re, bisect, datetime as dt, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ, CITY_RE, EXCL
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
EDGE=0.05; ENTRY_H=6; STAKE=50; TRAIN_END=dt.date(2026,6,30)
fc=pd.read_csv('out/forecast_pm2.csv')   # has fc_max/actual per (city,day) from the previous run (dedupe)
site=json.load(open('out/stations.json'))
fd=fc.drop_duplicates(['city','day'])[['city','unit','day','fc_max','actual']].copy(); fd['day']=pd.to_datetime(fd.day).dt.date
fd['err']=fd.actual-fd.fc_max
ms=[m for m in json.load(open('data/markets.json')) if CITY_RE.match(m['question'] or '') and not EXCL.search(m['question'])]
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    return (int(mm.group(4)),int(mm.group(4)))
ev={}
for m in ms:
    city=CITY_RE.match(m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); ev.setdefault((city,d),[]).append((m,bucket(m['question'])))
def yes_price(m,ts):
    try: h=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    if not h['history']: return None
    hs=sorted((x['t'],x['p']) for x in h['history']); i=bisect.bisect_right([t for t,_ in hs],ts)-1
    if i<0: return None
    p=hs[i][1]; return p if h['losing_outcome']=='Yes' else 1-p
rows=[]
for city,g in fd.groupby('city'):
    tr=g[g.day<=TRAIN_END]; te=g[g.day>TRAIN_END]
    if len(tr)<60 or len(te)==0: continue
    dist=tr.err.value_counts(normalize=True)          # P(err) per station, learned Jan-Jun
    tz=ZoneInfo(TZ[city])
    for r in te.itertuples():
        if (city,r.day) not in ev: continue
        ts=int(dt.datetime(r.day.year,r.day.month,r.day.day,ENTRY_H,tzinfo=tz).timestamp())
        for m,b in ev[(city,r.day)]:
            lo,hi=b; p_model=float(sum(v for e,v in dist.items() if lo<=r.fc_max+e<=hi))
            px=yes_price(m,ts)
            if px is None: continue
            won=[float(x) for x in m['outcome_prices']]==[1.0,0.0]
            rows.append(dict(city=city,unit=r.unit,day=str(r.day),bucket=m['question'].split(' be ',1)[1].split(' on')[0],p_model=p_model,px=px,edge=p_model-px,won=won,train_n=len(tr)))
df=pd.DataFrame(rows); df.to_csv('out/forecast_yes.csv',index=False)
# calibration of the model itself
df['pb']=pd.cut(df.p_model,[-.01,.05,.1,.2,.3,.4,.6,1.0])
print("model calibration on TEST (Jul-Sep): predicted vs realized bucket win rate")
print(df.groupby('pb',observed=True).apply(lambda g: pd.Series(dict(n=len(g),pred=round(g.p_model.mean(),3),realized=round(g.won.mean(),3),mkt_px=round(g.px.mean(),3)))).to_string())
for edge in [0.0,0.05,0.10,0.15]:
    t=df[(df.edge>=edge)&(df.px>=0.02)&(df.px<=0.6)].copy(); t['fill']=(t.px+0.005).clip(upper=.999); t['sh']=STAKE/t.fill; t['fee']=t.sh*0.05*t.fill*(1-t.fill); t['pnl']=np.where(t.won,t.sh,0)-STAKE-t.fee
    print(f"\nEDGE>={edge:.2f}: {len(t)} trades over {t.groupby(['city','day']).ngroups} city-days, avg px {t.px.mean():.3f}, model p {t.p_model.mean():.3f}, won {t.won.mean():.1%}, P&L ${t.pnl.sum():,.0f} on ${STAKE*len(t):,.0f} ({t.pnl.sum()/(STAKE*len(t)):+.1%})")
    if edge==0.05: print(t.groupby('unit').apply(lambda g: f"n={len(g)} won={g.won.mean():.1%} roi={g.pnl.sum()/(STAKE*len(g)):+.1%}").to_dict()); print(t.groupby('city').pnl.sum().round(0).sort_values().to_dict())
