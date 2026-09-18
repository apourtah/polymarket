import pandas as pd, numpy as np, json, re, bisect, datetime as dt
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ, CITY_RE
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
t=pd.read_parquet('out/trader_all_trades.parquet'); f=pd.read_parquet('out/trader_forecasts.parquet')
site=json.load(open('out/stations.json')); ms=[m for m in json.load(open('data/markets.json')) if CITY_RE.match(m['question'] or '')]
ev={}
for m in ms:
    city=CITY_RE.match(m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); ev.setdefault((city,d),[]).append(m)
def lo_of(q):
    mm=re.search(r'between (-?\d+)-|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC] on',q); return float(mm.group(1) or mm.group(2) or mm.group(4))
cache={}
def yes_series(m):
    if m['market_id'] not in cache:
        try: h=json.load(open(f"data/prices/{m['market_id']}.json")); hs=sorted((x['t'],x['p']) for x in h['history']); cache[m['market_id']]=([a for a,_ in hs],[(p if h['losing_outcome']=='Yes' else 1-p) for _,p in hs])
        except Exception: cache[m['market_id']]=None
    return cache[m['market_id']]
def fav_at(city,d,ts):
    best=None
    for m in ev.get((city,d),[]):
        if re.search(r'or (higher|below)',m['question']): continue
        s=yes_series(m)
        if not s: continue
        i=bisect.bisect_right(s[0],ts)-1
        if i<0: continue
        if best is None or s[1][i]>best[1]: best=(lo_of(m['question']),s[1][i])
    return best
def mid_at(city,d,lo,ts):
    for m in ev.get((city,d),[]):
        if lo_of(m['question'])==lo and not re.search(r'or (higher|below)',m['question']):
            s=yes_series(m)
            if not s: return None
            i=bisect.bisect_right(s[0],ts)-1; return s[1][i] if i>=0 else None
met=pd.read_parquet('data/metar.parquet'); met['day']=met.local_time.dt.date; met['tf']=(met.temp_c*9/5+32).map(rh)
mg={k:v.sort_values('local_time') for k,v in met.groupby(['station','day'])}
def obs_max(city,d,local_dt):
    g=mg.get((site[city][0],d))
    if g is None: return None,None
    g2=g[g.local_time<=local_dt.replace(tzinfo=None)]; return (int(g2.tf.max()) if len(g2) else None), int(g.tf.max())
fp=f.pivot_table(index=['city','mday'],columns='model',values='fmax')
rows=[]
for r in t.itertuples():
    ts=int(r.t.timestamp()); fav=fav_at(r.city,r.mday,ts); mid=mid_at(r.city,r.mday,r.lo,ts)
    om,fm=obs_max(r.city,r.mday,r.local)
    rec=dict(city=r.city,mday=r.mday,t=r.t,local=r.local.strftime('%m-%d %H:%M'),lhour=r.lhour,rel_day=r.rel_day,side=r.outcome,lo=r.lo,price=r.price,usd=r.usd,pnl=r.pnl,won=(r.pnl>0) if pd.notna(r.pnl) else None,
             fav_lo=fav[0] if fav else None,fav_p=fav[1] if fav else None,mid=mid,obs_max=om,final_max=fm)
    if fav: rec['off_fav']=(r.lo-fav[0])/2
    rec['bucket_won']=(r.pnl>0)==(r.outcome=='Yes') if pd.notna(r.pnl) else None
    for mdl in fp.columns:
        v=fp.loc[(r.city,r.mday),mdl] if (r.city,r.mday) in fp.index else np.nan
        rec[mdl]=v; rec[f'{mdl}_off']=((rh(v)-(rh(v)%2))-r.lo)/2 if pd.notna(v) else np.nan   # model bucket minus their bucket, in buckets
    rows.append(rec)
d=pd.DataFrame(rows); d.to_csv('out/trader_deep.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
res=d[d.pnl.notna()]
print(f"{len(d)} fills, {d.groupby(['city','mday']).ngroups} city-days, P&L ${res.pnl.sum():,.0f} on ${res.usd.sum():,.0f}\n")
print("=== 1. WHAT they buy, relative to the market favorite at that moment (offset in 2°F buckets; + = warmer than favorite)")
print(pd.crosstab(res.side, res.off_fav.clip(-3,3), values=res.usd, aggfunc='sum').round(0).fillna(0).to_string())
print("P&L by side x offset:"); print(pd.crosstab(res.side, res.off_fav.clip(-3,3), values=res.pnl, aggfunc='sum').round(0).fillna(0).to_string())
print("win rate (bucket won) by side x offset:"); print(pd.crosstab(res.side, res.off_fav.clip(-3,3), values=res.bucket_won, aggfunc='mean').round(2).to_string())
print("\n=== 2. Price paid vs midpoint at that moment (taker pays above mid, maker below)")
res2=res[res.mid.notna()].copy(); res2['edge_c']=np.where(res2.side=='Yes',res2.price-res2.mid,res2.price-(1-res2.mid))*100
print(res2.groupby('side').edge_c.describe()[['count','mean','25%','50%','75%']].round(2).to_string())
print("\n=== 3. WHEN: fills by local hour and day-relative, with P&L")
print(pd.crosstab([res.rel_day],pd.cut(res.lhour,[-1,3,6,9,12,15,18,23]),values=res.pnl,aggfunc='sum').round(0).fillna(0).to_string())
print(pd.crosstab([res.rel_day],pd.cut(res.lhour,[-1,3,6,9,12,15,18,23]),values=res.usd,aggfunc='sum').round(0).fillna(0).to_string())
print("\n=== 4. Do they follow a model? YES-bucket vs each model's day-ahead max bucket (0 = same bucket)")
y=res[res.side=='Yes']
for mdl in [c for c in d.columns if c.endswith('_off')]:
    v=y[mdl].dropna(); print(f"  {mdl:<24} same bucket {(v==0).mean():.0%}   model 1 warmer {(v==1).mean():.0%}   model 1 cooler {(v==-1).mean():.0%}   n={len(v)}")
print("NO-bucket vs model bucket:")
n=res[res.side=='No']
for mdl in [c for c in d.columns if c.endswith('_off')]:
    v=n[mdl].dropna(); print(f"  {mdl:<24} same bucket {(v==0).mean():.0%}   model 1 warmer {(v==1).mean():.0%}   model 1 cooler {(v==-1).mean():.0%}")
print("\n=== 5. Their YES bucket vs the ACTUAL winner, and vs the models' accuracy on the same days")
yd=y.drop_duplicates(['city','mday']); print(f"their YES bucket won: {yd.bucket_won.mean():.0%} of {len(yd)} city-days (avg price {y.price.mean():.2f})")
for mdl in ['ecmwf_ifs025_d1','gfs_seamless_d1','icon_seamless_d1','gem_seamless_d1']:
    v=yd.dropna(subset=[mdl]); hit=((v[mdl].map(rh)-v[mdl].map(rh)%2)==(v.final_max-v.final_max%2)).mean(); print(f"  {mdl} bucket = winner on same days: {hit:.0%}")
print(f"  market favorite (at their first fill) = winner: {(yd.fav_lo==yd.final_max-yd.final_max%2).mean():.0%}")
