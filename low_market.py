"""Market side of the 'lowest temperature' study, 8 US cities: bucket YES prices at 21:00 local the evening before,
favorite hit rate & calibration, compared with the same cities' 'highest temperature' markets over the same days."""
import json, re, bisect, datetime as dt, pandas as pd, numpy as np
from zoneinfo import ZoneInfo
from sweep_local import TZ
site=json.load(open('out/stations.json')); rh=lambda c: int(np.floor(c*9/5+32+0.5))
US=['New York City','Miami','Atlanta','Austin','Houston','Los Angeles','Seattle','San Francisco']
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
def price_at(m, ts):
    try: h=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    hs=h['history']
    if not hs: return None
    i=bisect.bisect_right([x['t'] for x in hs],ts)-1
    if i<0 or ts-hs[i]['t']>3*3600: return None
    p=hs[i]['p']; return p if h['losing_outcome']=='Yes' else 1-p      # history is the losing token's price -> YES price
def panel(kind, hour=21):
    ms=[m for m in json.load(open('data/markets.json')) if re.search(rf'{kind} temperature in ({"|".join(US)}) be ', m['question'] or '') and m.get('closed')]
    rows=[]
    for m in ms:
        city=re.search(r'temperature in (.+?) be ',m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); b=bucket(m['question'])
        ts=int(dt.datetime(d.year,d.month,d.day,hour,tzinfo=ZoneInfo(TZ[city])).timestamp())-86400
        p=price_at(m,ts); won=[float(x) for x in m['outcome_prices']]==[1.0,0.0]
        rows.append(dict(city=city,day=d,lo=b[0],hi=b[1],price=p,won=won,vol=float(m.get('volume') or 0),q=m['question']))
    return pd.DataFrame(rows)
if __name__=='__main__':
    pd.set_option('display.width',200)
    out={}
    for kind in ('lowest','highest'):
        P=panel(kind); P.to_parquet(f'out/{kind}_market_panel.parquet')
        # keep city-days with a single YES winner and prices for >= 3 buckets
        g=P.groupby(['city','day']); ok=g.won.transform('sum')==1; np_=g.price.transform('count')>=3; P=P[ok&np_]
        days=P.groupby(['city','day'])
        fav=P.loc[days.price.idxmax()]
        print(f"\n=== {kind}: {fav.groupby('city').size().to_dict()} city-days (from {P.day.min()} to {P.day.max()})")
        s=fav.groupby('city').agg(n=('won','size'),fav_price=('price','mean'),fav_win=('won','mean'),day_vol=('vol',lambda v: 0))
        s['day_vol']=P.groupby(['city','day']).vol.sum().groupby('city').mean().round(0)
        s.loc['ALL']=[len(fav),fav.price.mean(),fav.won.mean(),P.groupby(['city','day']).vol.sum().mean()]
        print(s.round(3).to_string())
        # calibration: YES win rate by price bin at 21:00 local evening before
        P['bin']=pd.cut(P.price,[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0])
        cal=P.groupby('bin',observed=True).agg(n=('won','size'),price=('price','mean'),win=('won','mean')); cal['edge_for_YES']=cal.win-cal.price
        print(cal.round(3).to_string())
        out[kind]=P
    # lows: where in the day does the min occur? (METAR)
    met=pd.read_parquet('data/metar.parquet'); met=met[met.city.isin(US)&(met.local_time.dt.date>=dt.date(2026,5,25))].copy(); met['day']=met.local_time.dt.date
    idx=met.groupby(['city','day']).temp_c.idxmin(); tmin=met.loc[idx]; tmin['h']=tmin.local_time.dt.hour
    print("\nhour of the daily MIN (local), share of days:")
    print(pd.crosstab(tmin.city,pd.cut(tmin.h,[-1,3,7,11,17,20,23],labels=['00-03','04-07','08-11','12-17','18-20','21-23']),normalize='index').round(2).to_string())
