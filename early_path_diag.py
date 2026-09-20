"""Does the early path (hours after the 21:35 entry) predict the outcome? Diagnostics before designing a time-armed exit:
 1. mean price change from entry by hours since entry, winners vs losers
 2. win rate conditional on the change at h hours (is 'below entry at +6h' informative?)
 3. volatility (5-min |dp|) by hour since entry — where is the noise?
 4. simple time-conditional exits: at hour h, sell if pos < entry + c (else hold); vs hold"""
import pandas as pd, numpy as np, json, bisect, datetime as dt
from zoneinfo import ZoneInfo
from sweep_local import TZ
pd.set_option('display.width',250)
exec(open('trail_stop_backtest.py').read().split("TRAILS=")[0])
paths=[]
for r in T.itertuples():
    if r.market_id is None: continue
    try: h=json.load(open(f"data/prices/{r.market_id}.json"))
    except FileNotFoundError: continue
    hs=h['history']
    if not hs: continue
    ts=np.array([x['t'] for x in hs]); p=np.array([x['p'] for x in hs]); yes=p if h['losing_outcome']=='Yes' else 1-p
    tz=ZoneInfo(TZ[r.city]); t0=int(dt.datetime(r.mday.year,r.mday.month,r.mday.day,21,35,tzinfo=tz).timestamp())-86400; i0=bisect.bisect_right(ts,t0)
    pos=yes[i0:] if r.side=='yes' else 1-yes[i0:]; hrs=(ts[i0:]-t0)/3600
    if len(pos)<2: continue
    entry=(r.price if r.side=='yes' else 1-r.price)+SLIP; won=r.won if r.side=='yes' else (not r.won)
    paths.append(dict(pos=pos,hrs=hrs,entry=entry,won=won,leg=r.leg,side=r.side))
print(len(paths),"paths (entry 21:35 local; +12h = 09:35 local market day, +18h = 15:35, +21h = 18:35)")
HS=[1,2,3,4,6,8,10,12,14,16,18,20,22]
def at(P,h):
    i=np.searchsorted(P['hrs'],h,side='right')-1; return P['pos'][i] if i>=0 else np.nan
rows=[]
for h in HS:
    d=pd.DataFrame([dict(dp=at(P,h)-P['entry'],won=P['won'],entry=P['entry']) for P in paths]).dropna()
    rows.append(dict(h=h,n=len(d),mean_dp_winners=d[d.won].dp.mean(),mean_dp_losers=d[~d.won].dp.mean(),win_if_below_entry=d[d.dp<0].won.mean(),n_below=(d.dp<0).sum(),win_if_above=d[d.dp>=0].won.mean(),win_if_below_m10=d[d.dp<-0.10].won.mean(),n_below_m10=(d.dp<-0.10).sum(),win_if_above_p10=d[d.dp>=0.10].won.mean(),base=d.won.mean()))
print("\n### 1-2. price change since entry by hours since entry, and win rate conditional on where the price is at that hour"); print(pd.DataFrame(rows).round(3).to_string(index=False))
# 3. volatility by hour since entry
v=[]
for P in paths:
    dp=np.abs(np.diff(P['pos'])); hh=P['hrs'][1:]
    for a,b in zip(HS[:-1],HS[1:]):
        m=(hh>=a)&(hh<b)
        if m.any(): v.append(dict(h=f"{a}-{b}",vol=dp[m].mean(),moves10=(dp[m]>=0.10).mean()))
V=pd.DataFrame(v).groupby('h').agg(mean_abs_5min_move=('vol','mean'),share_moves_ge10c=('moves10','mean')); V=V.loc[[f"{a}-{b}" for a,b in zip(HS[:-1],HS[1:])]]; print("\n### 3. 5-min volatility by hours since entry"); print(V.round(4).to_string())
# 4. time-conditional exits
def ev(P,h,c,hold_only=False):
    entry=P['entry']; sh=STAKE/entry; fee_in=0.05*entry*(1-entry)*sh; hold=sh*(1.0 if P['won'] else 0.0)-STAKE-fee_in
    if hold_only: return hold
    px=at(P,h)
    if np.isnan(px) or px>=entry+c or px<=0.01: return hold
    x=max(px-SLIP,0.005); return sh*x-STAKE-fee_in-0.05*x*(1-x)*sh
H=sum(ev(P,0,0,True) for P in paths); rows=[]
for h in [3,6,9,12,15,18]:
    for c in [-0.15,-0.10,-0.05,0.0,0.05]:
        pn=[ev(P,h,c) for P in paths]; ex=sum(1 for P in paths if not np.isnan(at(P,h)) and at(P,h)<P['entry']+c)
        rows.append(dict(exit_hour=h,sell_if_below_entry_plus=c,exited=ex/len(paths),pnl=sum(pn),vs_hold=sum(pn)-H,win_rate=np.mean(np.array(pn)>0)))
print(f"\n### 4. 'at hour h since entry, sell if the price is below entry+c, else hold' (hold = ${H:+,.0f})"); print(pd.DataFrame(rows).round(3).to_string(index=False))
# early reversal: rose >= +x by hour h1 then back below entry by h2
rows=[]
for x in [0.05,0.10]:
    for h1,h2 in [(3,6),(6,9),(6,12),(9,12)]:
        d=[]
        for P in paths:
            m1=P['hrs']<=h1; m2=(P['hrs']>h1)&(P['hrs']<=h2)
            if not m1.any() or not m2.any(): continue
            rose=P['pos'][m1].max()>=P['entry']+x; back=P['pos'][m2].min()<P['entry']
            d.append((rose,back,P['won']))
        d=pd.DataFrame(d,columns=['rose','back','won']); s=d[d.rose&d.back]; r_=d[d.rose&~d.back]
        rows.append(dict(rise=x,by_h=h1,back_below_by_h=h2,n_rose=int(d.rose.sum()),n_rose_then_back=len(s),win_rose_then_back=s.won.mean(),win_rose_held=r_.won.mean(),win_never_rose=d[~d.rose].won.mean()))
print("\n### early reversal: price rose >= x above entry within h1 hours, then fell back below entry by h2"); print(pd.DataFrame(rows).round(3).to_string(index=False))

# 5. calibration of the mid at hour h for OUR positions: if realised win rate < mid, selling has edge; if > mid, holding does
print("\n### 5. calibration at hour h: realised win rate vs the mid of our position at that hour (positive gap = holding is +EV)")
rows=[]
for h in [3,6,9,12,15,18]:
    d=pd.DataFrame([dict(mid=at(P,h),won=P['won'],entry=P['entry']) for P in paths]).dropna(); d['band']=pd.cut(d.mid,[0,.15,.3,.45,.6,.8,1.0])
    g=d.groupby('band',observed=True).agg(n=('won','size'),mid=('mid','mean'),win=('won','mean')); g['gap']=g.win-g.mid; g['h']=h; rows.append(g.reset_index())
R=pd.concat(rows); print(R.pivot(index='band',columns='h',values='gap').round(3).to_string()); print("\ncounts:"); print(R.pivot(index='band',columns='h',values='n').to_string())
d=pd.DataFrame([dict(mid=at(P,12),won=P['won'],entry=P['entry'],below=at(P,12)<P['entry']) for P in paths]).dropna()
for k,g in d.groupby('below'): print(f"at +12h, {'below' if k else 'at/above'} entry: n={len(g)}, mean mid {g.mid.mean():.3f}, realised win {g.won.mean():.3f}, gap {g.won.mean()-g.mid.mean():+.3f}")
