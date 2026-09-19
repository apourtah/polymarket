"""Test the 0xdd22-style NO leg on our backtest nights: NO on the market favorite when our model bucket differs.
Uses out/backtest_evening_nights.parquet (be/br/win_lo per city-night) + data/prices for every bucket."""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, os
from zoneinfo import ZoneInfo
from sweep_local import TZ
pd.set_option('display.width',250)
N=pd.read_parquet('out/backtest_evening_nights.parquet'); N['mday']=pd.to_datetime(N.mday).dt.date
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
cities=sorted(N.city.unique())
ms=[m for m in json.load(open('data/markets.json')) if re.search(rf'highest temperature in ({"|".join(cities)}) be ', m['question'] or '') and m.get('closed')]
mk={}
for m in ms:
    city=re.search(r'temperature in (.+?) be ',m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); mk.setdefault((city,d),{})[bucket(m['question'])]=m
cache={}
def yes_price(m, ts):
    mid=m['market_id']
    if mid not in cache:
        try: h=json.load(open(f"data/prices/{mid}.json")); cache[mid]=([x['t'] for x in h['history']],[x['p'] if h['losing_outcome']=='Yes' else 1-x['p'] for x in h['history']])
        except FileNotFoundError: cache[mid]=None
    s=cache[mid]
    if not s or not s[0]: return None
    i=bisect.bisect_right(s[0],ts)-1
    if i<0 or ts-s[0][i]>2*3600: return None
    return s[1][i]
rows=[]
for r in N.itertuples():
    bk=mk.get((r.city,r.mday));
    if not bk: continue
    tz=ZoneInfo(TZ[r.city]); d=r.mday
    for label,ts in [('21:35 eve',int(dt.datetime(d.year,d.month,d.day,21,35,tzinfo=tz).timestamp())-86400),('03:00 day',int(dt.datetime(d.year,d.month,d.day,3,0,tzinfo=tz).timestamp())),('06:30 day',int(dt.datetime(d.year,d.month,d.day,6,30,tzinfo=tz).timestamp()))]:
        pr={b:yes_price(m,ts) for b,m in bk.items()}; pr={b:v for b,v in pr.items() if v is not None}
        if len(pr)<3: continue
        fav=max(pr,key=pr.get); fp=pr[fav]
        rows.append(dict(city=r.city,mday=d,when=label,agree=r.agree,be=r.be,br=r.br,win_lo=r.win_lo,fav=fav[0],fav_p=fp,fav_won=(fav[0]==r.win_lo),
                         off_e=(fav[0]-r.be)/2 if fav[0]>-999 else np.nan, p_be=pr.get((r.be,r.be+1)), p_br=pr.get((r.br,r.br+1))))
D=pd.DataFrame(rows); D.to_parquet('out/no_leg_nights.parquet')
def sim(x,slip=0.01,stake=50):
    # buy NO on the favorite: NO ask = 1 - fav_p + slip
    ask=(1-x.fav_p+slip).clip(0.01,0.99); sh=stake/ask; fee=0.05*ask*(1-ask)*sh; won=~x.fav_won
    pnl=np.where(won,sh-stake,-stake)-fee; return pd.Series(dict(n=len(x),fav_win=x.fav_won.mean(),no_px=ask.mean(),stake=stake*len(x),pnl=pnl.sum(),roi=pnl.sum()/(stake*len(x)) if len(x) else np.nan))
for when in ['21:35 eve','03:00 day','06:30 day']:
    X=D[D.when==when].copy(); X['dis_e']=X.fav!=X.be; X['dis_r']=X.fav!=X.br; X['dis_both']=X.dis_e&X.dis_r
    print(f"\n===== entry {when}: {len(X)} nights, favorite wins {X.fav_won.mean():.0%} (fav price mean {X.fav_p.mean():.2f})")
    print("NO on favorite, ALL nights, by favorite price band:"); print(X.groupby(pd.cut(X.fav_p,[0,.3,.4,.5,.6,.7,1]),observed=True).apply(sim).round(2).to_string())
    for cond,name in [('dis_e','fav != EWMA bucket'),('dis_r','fav != ridge bucket'),('dis_both','fav != both models')]:
        Y=X[X[cond]]; print(f"\nNO on favorite when {name} ({len(Y)} nights):"); print(Y.groupby(pd.cut(Y.fav_p,[0,.3,.4,.5,.6,.7,1]),observed=True).apply(sim).round(2).to_string())
    Y=X[X.dis_both]; print("\n  ...fav != both, by city (fav 0.40-0.70):"); print(Y[Y.fav_p.between(.4,.7)].groupby('city').apply(sim).round(2).to_string())
    Y=X[X.dis_both&X.fav_p.between(.4,.7)].copy(); Y['month']=pd.to_datetime(Y.mday).dt.to_period('M'); print("  ...by month:"); print(Y.groupby('month').apply(sim).round(2).to_string())
    print("  ...by offset (favorite − EWMA bucket, buckets):"); print(X[X.fav_p.between(.4,.7)].groupby(X.off_e.clip(-3,3)).apply(sim).round(2).to_string())
