"""Where does the day's winner land relative to the market favorite (06:00 local)? By city and month,
all stations. Then an out-of-sample test: use each city's bias measured on prior months to pick a side."""
import json, re, bisect, datetime as dt, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from sweep_local import TZ, CITY_RE
STAKE=50.0; site=json.load(open('out/stations.json'))
def lo_of(q):
    mm=re.search(r'between (-?\d+)-|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC] on',q); return float(mm.group(1) or mm.group(2) or mm.group(4))
def is_tail(q): return bool(re.search(r'or (higher|below)',q))
ms=[m for m in json.load(open('data/markets.json')) if CITY_RE.match(m['question'] or '')]
ev={}
for m in ms:
    city=CITY_RE.match(m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); ev.setdefault((city,d),[]).append(m)
def series(m):
    try: h=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    if not h['history']: return None
    hs=sorted((x['t'],x['p']) for x in h['history']); return ([t for t,_ in hs],[(p if h['losing_outcome']=='Yes' else 1-p) for _,p in hs])
def yes_at(m,ts):
    s=series(m)
    if not s: return None
    i=bisect.bisect_right(s[0],ts)-1; return s[1][i] if i>=0 else None
rows=[]
for (city,d),mk in ev.items():
    if city not in TZ or city not in site: continue
    unit=site[city][1]; step=2 if unit=='F' else 1
    win=[m for m in mk if [float(x) for x in m['outcome_prices']]==[1.0,0.0]]
    if len(win)!=1: continue
    ts=int(dt.datetime(d.year,d.month,d.day,6,tzinfo=ZoneInfo(TZ[city])).timestamp())
    pr=[(m,yes_at(m,ts)) for m in mk if not is_tail(m['question'])]; pr=[(m,p) for m,p in pr if p is not None]
    if not pr: continue
    fav,pf=max(pr,key=lambda x:x[1]); flo=lo_of(fav['question']); off=(lo_of(win[0]['question'])-flo)/step
    byl={lo_of(m['question']):(m,p) for m,p in pr}
    rec=dict(city=city,unit=unit,day=d,month=d.strftime('%Y-%m'),fav_prob=pf,off=off)
    for name,k in (('warm',flo+step),('cool',flo-step)):
        if k in byl: rec[f'px_{name}']=byl[k][1]; rec[f'won_{name}']=(off==(1 if name=='warm' else -1))
    rows.append(rec)
df=pd.DataFrame(rows); df.to_csv('out/favbias.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
print(f"{len(df)} city-days, {df.city.nunique()} cities")
print("\nOverall: winner vs favorite:", df.off.clip(-2,2).value_counts(normalize=True).sort_index().round(3).to_dict())
print("by unit:"); print(df.groupby('unit').apply(lambda g: pd.Series(dict(n=len(g),cooler=(g.off<0).mean(),fav=(g.off==0).mean(),warmer=(g.off>0).mean(),net_warm=(g.off>0).mean()-(g.off<0).mean()))).round(3).to_string())
# city x month: net warm skew = P(warmer) - P(cooler)
piv=df.groupby(['city','month']).apply(lambda g:(g.off>0).mean()-(g.off<0).mean()).unstack('month').round(2)
n=df.groupby('city').size(); piv['ALL']=df.groupby('city').apply(lambda g:(g.off>0).mean()-(g.off<0).mean()).round(2); piv['n']=n
print("\nNET WARM SKEW = P(winner warmer than favorite) − P(cooler), city (rows) x month; + = days run warmer than the market's morning pick")
print(piv.sort_values('ALL',ascending=False).to_string())
print("\nby month, all cities:"); print(df.groupby('month').apply(lambda g: pd.Series(dict(n=len(g),cooler=(g.off<0).mean(),fav=(g.off==0).mean(),warmer=(g.off>0).mean()))).round(3).to_string())
# --- out-of-sample adaptive test: for month M, per city, use all prior months' skew; if |skew|>=TH buy YES on that side
def pnl(px,won):
    fill=min(px+0.005,0.999); sh=STAKE/fill; fee=sh*0.05*fill*(1-fill); return (sh if won else 0)-STAKE-fee
months=sorted(df.month.unique()); out=[]
for TH in [0.05,0.10,0.15,0.20]:
    for M in months[2:]:
        hist=df[df.month<M]; cur=df[df.month==M]
        skew=hist.groupby('city').apply(lambda g:(g.off>0).mean()-(g.off<0).mean()); cnt=hist.groupby('city').size()
        for r in cur.itertuples():
            s=skew.get(r.city,0); 
            if cnt.get(r.city,0)<30 or abs(s)<TH: continue
            side='warm' if s>0 else 'cool'; px=getattr(r,f'px_{side}',None); won=getattr(r,f'won_{side}',None)
            if px is None or np.isnan(px) or px<0.02 or px>0.8: continue
            out.append(dict(TH=TH,month=M,city=r.city,side=side,px=px,won=bool(won),pnl=pnl(px,bool(won))))
o=pd.DataFrame(out)
print("\nOUT-OF-SAMPLE: buy $50 YES on the side each city skewed to in all PRIOR months (|skew| >= TH), scored on the current month")
print(o.groupby('TH').apply(lambda g: pd.Series(dict(trades=len(g),warm=(g.side=='warm').mean(),avg_px=g.px.mean(),win=g.won.mean(),breakeven=(g.px+0.005).mean(),pnl=g.pnl.sum(),roi=g.pnl.sum()/(STAKE*len(g))))).round(3).to_string())
print("\nTH=0.10 by month:"); print(o[o.TH==0.10].groupby('month').apply(lambda g: pd.Series(dict(trades=len(g),win=g.won.mean(),pnl=g.pnl.sum(),roi=g.pnl.sum()/(STAKE*len(g))))).round(3).to_string())
