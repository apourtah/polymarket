"""Can we calibrate size/participation to model confidence? On the 8-month dump: split the production trades by
 (a) the city's trailing HRRR |error| (5- and 10-day means, lagged = known at decision time),
 (b) the models' own bucket probability for the pick (avg P),
 (c) the models' recent residual SD proxy = width implied by P (via pav),
and test simple gates / size scalings."""
import pandas as pd, numpy as np
pd.set_option('display.width',250)
exec(open('trail_stop_backtest.py').read().split("# market ids")[0])
H=pd.read_parquet('data/evening_history.parquet'); H['mday']=pd.to_datetime(H.mday); H=H.sort_values(['city','mday'])
H['ae']=H.err.abs(); H['mae5']=H.groupby('city').ae.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean()); H['mae10']=H.groupby('city').ae.transform(lambda s: s.shift(1).rolling(10,min_periods=5).mean())
H['pool_mae5']=H.groupby('mday').ae.transform('mean'); H['pool_mae5']=H.groupby('city').pool_mae5.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean())   # all-city recent error (regime)
T=T.merge(H[['city','mday','mae5','mae10','pool_mae5']],on=['city','mday'],how='left'); T['pav']=(T.pe+T.pr)/2; T['month']=T.mday.dt.to_period('M').astype(str)
ask=np.where(T.side=='yes',T.price+SLIP,1-T.price+SLIP); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(T.side=='yes',T.won,~T.won); T['pnl']=np.where(win,sh-STAKE,-STAKE)-fee; T['win']=win
def summ(x):
    if len(x)==0: return pd.Series(dict(n=0))
    m=x.groupby('month').pnl.sum(); return pd.Series(dict(n=len(x),win=x.win.mean(),roi=x.pnl.sum()/(STAKE*len(x)),pnl=x.pnl.sum(),neg_m=(m<0).sum()))
print(f"{len(T)} trades, hold P&L ${T.pnl.sum():+,.0f}\n")
print("### (a) by the city's trailing 5-day HRRR |error| (known at decision time)"); print(T.groupby(pd.cut(T.mae5,[0,1.5,2,2.5,3,4,10]),observed=True).apply(summ).round(2).to_string())
print("\n    by trailing 10-day |error|:"); print(T.groupby(pd.cut(T.mae10,[0,1.5,2,2.5,3,4,10]),observed=True).apply(summ).round(2).to_string())
print("\n    by the ALL-city trailing 5-day |error| (regime):"); print(T.groupby(pd.cut(T.pool_mae5,[0,2,2.5,3,3.5,10]),observed=True).apply(summ).round(2).to_string())
print("\n### (b) by the models' own probability for the pick (avg of EWMA/ridge P), YES legs"); Y=T[T.side=='yes']; print(Y.groupby(pd.cut(Y.pav,[0,.15,.25,.35,.45,.6,1]),observed=True).apply(summ).round(2).to_string())
print("\n    model P minus price (edge), YES legs:"); print(Y.groupby(pd.cut(Y.pav-Y.price,[-1,-.15,-.05,.05,.15,1]),observed=True).apply(summ).round(2).to_string())
print("\n### gates (skip the trade when ...):")
rows=[]
for name,mask in [('none',T.mae5.notna()|T.mae5.isna()),('city mae5 <= 3',~(T.mae5>3)),('city mae5 <= 2.5',~(T.mae5>2.5)),('city mae10 <= 3',~(T.mae10>3)),('pool mae5 <= 3',~(T.pool_mae5>3)),('pool mae5 <= 2.75',~(T.pool_mae5>2.75)),('YES pav >= 0.25',(T.side=='no')|(T.pav>=0.25)),('YES pav >= 0.30',(T.side=='no')|(T.pav>=0.30)),('city mae5<=3 & pav>=0.25',~(T.mae5>3)&((T.side=='no')|(T.pav>=0.25)))]:
    x=T[mask]; s=summ(x); s['kept']=len(x)/len(T); d=x.groupby('mday').pnl.sum().sort_index().cumsum(); s['max_dd']=(d-d.cummax()).min(); rows.append(pd.Series(s,name=name))
print(pd.DataFrame(rows).round(2).to_string())
print("\n### size scaling instead of gating: stake x f(mae5): 1 if <=2, 0.5 if 2-3, 0.25 if >3")
w=np.select([T.mae5<=2,T.mae5<=3],[1.0,0.5],0.25); w=np.where(np.isnan(T.mae5),1.0,w); print(f"scaled: P&L ${(T.pnl*w).sum():+,.0f} on ${(STAKE*w).sum():,.0f} -> ROI {(T.pnl*w).sum()/(STAKE*w).sum():+.1%} (flat: {T.pnl.sum()/(STAKE*len(T)):+.1%})")
