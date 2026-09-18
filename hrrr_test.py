"""Does HRRR (00Z run = evening before in the US; 12Z run = early morning) reproduce their bet?
For each city-day: HRRR afternoon max per run -> bucket. Compare to their YES pick and the winner;
trade it at evening-before (21:00) and 06:00 prices."""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from sweep_local import TZ, CITY_RE
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP)); b2=lambda v: rh(v)-rh(v)%2
h=pd.read_parquet('data/hrrr_t2m.parquet'); h=h[h.city!='ERR'].copy(); h['valid_utc']=pd.to_datetime(h.valid_utc,utc=True)
h['local']=[t.tz_convert(TZ[c]) for t,c in zip(h.valid_utc,h.city)]; h['mday']=[x.date() for x in h.local]; h['lh']=[x.hour for x in h.local]
h=h[h.lh.between(11,18)]
hm=h.groupby(['city','mday','run']).t2m_f.max().unstack('run').rename(columns={0:'hrrr00',12:'hrrr12'})
print("HRRR city-days:", len(hm), "| cities", hm.index.get_level_values(0).nunique())
a=pd.read_csv('out/trader_dayselect.csv'); a['mday']=pd.to_datetime(a.mday).dt.date; a=a[a.mday>=dt.date(2026,6,1)]
c=pd.read_csv('out/trader_citydays.csv'); c['mday']=pd.to_datetime(c.mday).dt.date; yes_pick=c.set_index(['city','mday']).yes_lo
a=a.join(hm,on=['city','mday']); a['their_yes']=[yes_pick.get((r.city,r.mday),np.nan) if (r.city,r.mday) in yes_pick.index else np.nan for r in a.itertuples()]
# HRRR bias vs station hourly max (in °F) -> calibrate a station offset from the first half, apply to second half
met=pd.read_parquet('data/metar.parquet'); site=json.load(open('out/stations.json')); met['day']=met.local_time.dt.date; met['tf']=met.temp_c*9/5+32
dmax=met.groupby(['station','day']).tf.max(); a['actual']=[dmax.get((site[r.city][0],r.mday),np.nan) for r in a.itertuples()]
for run in ['hrrr00','hrrr12']:
    e=(a.actual-a[run]).dropna(); print(f"{run}: station max − HRRR max: mean {e.mean():+.2f}°F, sd {e.std():.2f}, |err|<=1.5°F {(e.abs()<=1.5).mean():.0%}, n={len(e)}")
print("per-city bias (actual − hrrr12):", (a.actual-a.hrrr12).groupby(a.city).mean().round(2).to_dict())
their=sorted(c.city.unique())
ms=[m for m in json.load(open('data/markets.json')) if CITY_RE.match(m['question'] or '') and CITY_RE.match(m['question']).group(1) in their]
ev={}
for m in ms:
    city=CITY_RE.match(m['question']).group(1); dd=dt.date.fromisoformat(m['end_date'][:10]); ev.setdefault((city,dd),[]).append(m)
def lo_of(q):
    mm=re.search(r'between (-?\d+)-|(-?\d+)°[FC] or (higher|below)|(-?\d+)°[FC] on',q); return float(mm.group(1) or mm.group(2) or mm.group(4))
def prices_at(city,dd,hh,dayoff=0):
    ts=int(dt.datetime(dd.year,dd.month,dd.day,hh,tzinfo=ZoneInfo(TZ[city])).timestamp())+dayoff*86400; pr={}
    for m in ev.get((city,dd),[]):
        if re.search(r'or (higher|below)',m['question']): continue
        try: hh_=json.load(open(f"data/prices/{m['market_id']}.json"))
        except FileNotFoundError: continue
        if not hh_['history']: continue
        hs=sorted((x['t'],x['p']) for x in hh_['history']); i=bisect.bisect_right([q for q,_ in hs],ts)-1
        if i>=0: p=hs[i][1]; pr[lo_of(m['question'])]=p if hh_['losing_outcome']=='Yes' else 1-p
    return pr
STAKE=50
def pnl(px,won): fill=min(px+0.005,0.999); sh=STAKE/fill; return (sh if won else 0)-STAKE-sh*0.05*fill*(1-fill)
bias12=(a.actual-a.hrrr12).groupby(a.city).mean(); bias00=(a.actual-a.hrrr00).groupby(a.city).mean()
a['pick00']=[b2(v) if pd.notna(v) else np.nan for v in a.hrrr00]; a['pick12']=[b2(v) if pd.notna(v) else np.nan for v in a.hrrr12]
a['pick12c']=[b2(v+bias12.get(cty,0)) if pd.notna(v) else np.nan for v,cty in zip(a.hrrr12,a.city)]; a['pick00c']=[b2(v+bias00.get(cty,0)) if pd.notna(v) else np.nan for v,cty in zip(a.hrrr00,a.city)]
t=a[a.their_yes.notna()]
print(f"\n=== agreement with THEIR YES bucket on {len(t)} traded city-days")
for p in ['pick00','pick12','pick00c','pick12c']:
    v=t[t[p].notna()]; print(f"  {p:<8} same bucket as theirs {(v[p]==v.their_yes).mean():.0%} | HRRR hits winner {(v[p]==v.win_lo).mean():.0%} | theirs hits winner {(v.their_yes==v.win_lo).mean():.0%} | theirs vs HRRR offset: {((v.their_yes-v[p])/2).clip(-2,2).value_counts(normalize=True).sort_index().round(2).to_dict()}")
print("\n=== Trade it: $50 YES on the HRRR bucket (their 7 cities, all days Jun-Sep)")
for p,hh,off,label in [('pick00',21,-1,'00Z run @ 21:00 evening-before prices'),('pick00c',21,-1,'00Z bias-corrected @ 21:00'),('pick12',6,0,'12Z run @ 06:00 prices'),('pick12c',6,0,'12Z bias-corrected @ 06:00'),('pick12',9,0,'12Z run @ 09:00 prices'),('pick12c',9,0,'12Z bias-corrected @ 09:00')]:
    v=[];combo=[]
    for r in a.itertuples():
        lo=getattr(r,p)
        if pd.isna(lo): continue
        pr=prices_at(r.city,r.mday,hh,off)
        if not pr or lo not in pr or pr[lo]<0.02 or pr[lo]>0.9: continue
        v.append((pr[lo],r.win_lo==lo)); fav=max(pr,key=pr.get)
        if lo!=fav: combo.append(pnl(pr[lo],r.win_lo==lo)+pnl(1-pr[fav],r.win_lo!=fav))
    n=len(v); print(f"  {label:<40} n={n:>3} avg px {np.mean([q for q,_ in v]):.3f} win {np.mean([w for _,w in v]):.1%} breakeven {np.mean([min(q+.005,.999) for q,_ in v]):.1%} ROI {sum(pnl(q,w) for q,w in v)/(STAKE*n):+.1%} | +NO-fav combo ROI {sum(combo)/(2*STAKE*len(combo)):+.1%} (n={len(combo)})")
a.to_csv('out/hrrr_test.csv',index=False)
