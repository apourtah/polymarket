"""Cool-bias backtest (that account's structure): at ENTRY_H local on the market day, find the
market-favorite bucket (highest YES midpoint). Buy NO on it and YES on the bucket one step COOLER.
Also the mirror (warm) for comparison. $STAKE per leg, fill = mid + 0.5c, taker fee 0.05 p(1-p)."""
import json, re, bisect, datetime as dt, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from sweep_local import TZ, CITY_RE, EXCL
STAKE=50.0; ENTRY_H=int(__import__('os').environ.get('ENTRY_H','6'))
site=json.load(open('out/stations.json'))
def lo_of(q):
    mm=re.search(r'between (-?\d+)-|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC] on',q); return float(mm.group(1) or mm.group(2) or mm.group(4))
def kind(q): return 'tail' if re.search(r'or (higher|below)',q) else 'mid'
ms=[m for m in json.load(open('data/markets.json')) if CITY_RE.match(m['question'] or '')]
ev={}
for m in ms:
    city=CITY_RE.match(m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); ev.setdefault((city,d),[]).append(m)
def mid_at(m,ts,side):
    try: h=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    if not h['history']: return None
    hs=sorted((x['t'],x['p']) for x in h['history']); i=bisect.bisect_right([t for t,_ in hs],ts)-1
    if i<0: return None
    p=hs[i][1]; yes=p if h['losing_outcome']=='Yes' else 1-p; return yes if side=='Yes' else 1-yes
rows=[]
for (city,d),mk in ev.items():
    unit=site.get(city,(None,None))[1]
    if unit!='F': continue
    win=[m for m in mk if [float(x) for x in m['outcome_prices']]==[1.0,0.0]]
    if len(win)!=1: continue
    tz=ZoneInfo(TZ[city]); ts=int(dt.datetime(d.year,d.month,d.day,ENTRY_H,tzinfo=tz).timestamp())
    pr=[(m,mid_at(m,ts,'Yes')) for m in mk]; pr=[(m,p) for m,p in pr if p is not None and kind(m['question'])=='mid']
    if not pr: continue
    fav,pf=max(pr,key=lambda x:x[1]); flo=lo_of(fav['question']); wlo=lo_of(win[0]['question'])
    byl={lo_of(m['question']):m for m,_ in pr}
    legs=[('NO_fav',fav,'No',pf)]
    if flo-2 in byl: legs.append(('YES_cool',byl[flo-2],'Yes',mid_at(byl[flo-2],ts,'Yes')))
    if flo+2 in byl: legs.append(('YES_warm',byl[flo+2],'Yes',mid_at(byl[flo+2],ts,'Yes')))
    for name,m,side,p in legs:
        px=(p if side=='Yes' else 1-p)
        if px is None or px<=0.01 or px>=0.99: continue
        fill=min(px+0.005,0.999); sh=STAKE/fill; fee=sh*0.05*fill*(1-fill)
        bucket_won=lo_of(m['question'])==wlo; won=(side=='Yes')==bucket_won
        rows.append(dict(city=city,day=str(d),leg=name,px=px,won=won,pnl=(sh if won else 0)-STAKE-fee,fav_prob=pf,winner_vs_fav=(wlo-flo)/2))
df=pd.DataFrame(rows); df.to_csv(f'out/coolbias_h{ENTRY_H}.csv',index=False)
pd.set_option('display.width',220)
def S(g): return pd.Series(dict(n=len(g),avg_px=round(g.px.mean(),3),win=round(g.won.mean(),3),pnl=round(g.pnl.sum()),roi=round(g.pnl.sum()/(STAKE*len(g)),3)))
print(f"entry {ENTRY_H:02d}:00 local, US °F cities, {df.groupby(['city','day']).ngroups} city-days\n"); print(df.groupby('leg').apply(S).to_string())
print("\nwinner relative to the 06:00 favorite (buckets):", df[df.leg=='NO_fav'].winner_vs_fav.clip(-3,3).value_counts(normalize=True).sort_index().round(3).to_dict())
their=['Houston','Los Angeles','Atlanta','Miami','Austin','Seattle','San Francisco']
df['group']=np.where(df.city.isin(their),'their 7 cities','other US')
print("\nby group x leg:"); print(df.groupby(['group','leg']).apply(S).to_string())
print("\ncool combo (NO_fav + YES_cool) by city:"); c=df[df.leg.isin(['NO_fav','YES_cool'])].groupby('city').apply(S).sort_values('roi',ascending=False); print(c.to_string())
print("\ncool combo by month:"); print(df[df.leg.isin(['NO_fav','YES_cool'])].groupby(df.day.str[:7]).apply(S).to_string())
