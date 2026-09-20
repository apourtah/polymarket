"""Extended trailing-stop sweep on the production rule's 5-min paths. Dimensions:
 trail type: absolute cents below running max | percent of running max
 arm condition: gain >= A cents | position price >= L (lock near-wins) | local time on market day >= H | hours since entry >= T
 smoothing: raw mids | 30-min rolling median (6 samples) before testing the trail
 minimum hold: none | 6 h
Exit at (mid - 1c) with taker fee; unfilled -> hold. Reports P&L, win rate, max drawdown, months."""
import pandas as pd, numpy as np, json, bisect, datetime as dt, itertools
from zoneinfo import ZoneInfo
from sweep_local import TZ
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
    pos=yes[i0:] if r.side=='yes' else 1-yes[i0:]; tt=ts[i0:]
    if len(pos)<2: continue
    entry=(r.price if r.side=='yes' else 1-r.price)+SLIP; won=r.won if r.side=='yes' else (not r.won)
    lh=np.array([dt.datetime.fromtimestamp(t,tz).hour+dt.datetime.fromtimestamp(t,tz).minute/60 for t in tt]); sameday=np.array([dt.datetime.fromtimestamp(t,tz).date()==r.mday.date() for t in tt])
    paths.append(dict(pos=pos,hrs=(tt-t0)/3600,lh=lh,sameday=sameday,entry=entry,won=won,mday=r.mday,month=str(r.mday)[:7]))
print(f"{len(paths)} paths")
def smooth(x,w=6): return pd.Series(x).rolling(w,min_periods=1).median().values
def run(kind,val,arm,armv,sm,minhold):
    out=[]
    for P in paths:
        pos=P['pos']; entry=P['entry']; sh=STAKE/entry; fee_in=0.05*entry*(1-entry)*sh; final=1.0 if P['won'] else 0.0; hold=sh*final-STAKE-fee_in
        s=smooth(pos) if sm else pos; rm=np.maximum.accumulate(np.maximum(s,entry))
        trig=(s<=rm-val) if kind=='abs' else (s<=rm*(1-val))
        if arm=='gain': a=rm>=entry+armv
        elif arm=='level': a=rm>=armv
        elif arm=='lhour': a=P['sameday']&(P['lh']>=armv)
        else: a=P['hrs']>=armv
        if minhold: a=a&(P['hrs']>=minhold)
        t=trig&a; k=np.argmax(t) if t.any() else -1
        if k>=0 and 0.01<pos[k]<0.995:
            px=max(pos[k]-SLIP,0.005); out.append((sh*px-STAKE-fee_in-0.05*px*(1-px)*sh,1,P['mday']))
        else: out.append((hold,0,P['mday']))
    d=pd.DataFrame(out,columns=['pnl','stopped','mday']); day=d.groupby('mday').pnl.sum().sort_index(); eq=day.cumsum(); m=d.groupby(d.mday.dt.to_period('M')).pnl.sum()
    return dict(kind=kind,trail=val,arm=arm,arm_val=armv,smooth=sm,min_hold_h=minhold,stopped=d.stopped.mean(),win_rate=(d.pnl>0).mean(),pnl=d.pnl.sum(),roi=d.pnl.sum()/(STAKE*len(d)),max_dd=(eq-eq.cummax()).min(),neg_months=(m<0).sum(),worst_month=m.min())
H=sum((STAKE/P['entry'])*(1.0 if P['won'] else 0.0)-STAKE-0.05*P['entry']*(1-P['entry'])*(STAKE/P['entry']) for P in paths)
cfgs=[]
for kind,vals in [('abs',[0.10,0.20,0.30,0.40,0.50]),('pct',[0.20,0.35,0.50,0.65])]:
    for v in vals:
        for arm,avs in [('gain',[0.0,0.10,0.30]),('level',[0.60,0.75,0.90]),('lhour',[9,12,15]),('hours',[6,12])]:
            for av in avs:
                for sm in [False,True]:
                    for mh in [0,6]:
                        if arm in ('lhour','hours') and mh: continue
                        cfgs.append((kind,v,arm,av,sm,mh))
print(f"{len(cfgs)} configs; HOLD ${H:+,.0f}")
R=pd.DataFrame([run(*c) for c in cfgs]); R['vs_hold']=R.pnl-H; R.to_csv('out/trail_stop_sweep2.csv',index=False)
pd.set_option('display.width',250)
print("\n### top 15 by P&L"); print(R.sort_values('pnl',ascending=False).head(15).round(3).to_string(index=False))
print("\n### top 10 by P&L per $ of max drawdown (min P&L $8k)"); x=R[R.pnl>=8000].copy(); x['pnl_per_dd']=x.pnl/(-x.max_dd); print(x.sort_values('pnl_per_dd',ascending=False).head(10).round(3).to_string(index=False))
print("\n### best of each family"); print(R.sort_values('pnl',ascending=False).groupby(['kind','arm']).head(1).sort_values('pnl',ascending=False).round(3).to_string(index=False))
print(f"\nconfigs beating hold on P&L: {(R.pnl>H).sum()} of {len(R)}; with smaller max drawdown than hold AND within $1k of hold's P&L: {((R.max_dd>-903)&(R.pnl>H-1000)).sum()}")
