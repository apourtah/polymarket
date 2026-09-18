"""Verify the resolution rule for 'lowest temperature' markets: min over hourly/special METARs of the local day,
T-group tenths -> F, rounding half-up (or half-down?), vs the winning bucket."""
import json, re, datetime as dt, pandas as pd, numpy as np, collections
from sweep_local import TZ
site=json.load(open('out/stations.json'))
ms=[m for m in json.load(open('data/markets.json')) if re.search(r'lowest temperature', m['question'] or '') and m.get('closed')]
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
win={}
for m in ms:
    city=re.search(r'temperature in (.+?) be ',m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10])
    if [float(x) for x in m['outcome_prices']]==[1.0,0.0]: win[(city,d)]=bucket(m['question'])
print(len(win),"resolved city-days with a YES winner")
mt=pd.read_parquet('data/metar.parquet'); mt['day']=mt.local_time.dt.date
mins={(r.city,r.day):(r.temp_c) for r in mt.groupby(['city','day']).temp_c.min().reset_index().itertuples()}
def f_half_up(c): return int(np.floor(c*9/5+32+0.5))
def f_half_down(c): return int(np.ceil(c*9/5+32-0.5))
def f_floor(c): return int(np.floor(c*9/5+32))
def f_ceil(c): return int(np.ceil(c*9/5+32))
res=collections.defaultdict(collections.Counter)
miss=[]
for (city,d),b in win.items():
    if (city,d) not in mins or city not in site: continue
    c=mins[(city,d)]; unit=site[city][1]
    for name,fn in [('half_up',f_half_up),('half_down',f_half_down),('floor',f_floor),('ceil',f_ceil)] if unit=='F' else [('roundC',lambda c:int(np.floor(c+0.5)))]:
        v=fn(c); ok=b[0]<=v<=b[1]; res[(unit,name)][ok]+=1
        if name in('half_up','roundC') and not ok: miss.append((city,d,c,v,b))
for k,v in sorted(res.items()): print(k, dict(v), f"{v[True]/(v[True]+v[False]):.3f}")
print("misses (half-up / roundC):", len(miss)); 
for x in miss[:25]: print(x)
