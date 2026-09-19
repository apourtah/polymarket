"""Compare NO-leg rule variants on out/backtest_evening_buckets.parquet (7 cities, 21:35 mids, $50/leg, +1c, fee)."""
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
STAKE=50.0; SLIP=0.01
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B['month']=B.mday.dt.to_period('M').astype(str)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['modes']=B.city.map(lambda c: ','.join(sorted(MODES[c]))); B['is_fav']=B.lo==B.fav
Y=pd.concat([B[B.agree&B.is_be&B.modes.str.contains('agree')&B.price.between(0.10,0.50)].assign(why='agree'),
             B[~B.agree&B.is_br&B.modes.str.contains('ridge')&B.price.between(0.05,0.60)].assign(why='dis_ridge'),
             B[~B.agree&B.is_be&B.modes.str.contains('ewma')&B.price.between(0.05,0.60)].assign(why='dis_ewma')])
yk=Y.set_index(['city','mday']); B['yes_lo']=[yk.lo.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['ywhy']=[yk.why.get((c,d)) for c,d in zip(B.city,B.mday)]
def pnl_no(x): ask=(1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(~x.won,sh-STAKE,-STAKE)-fee
def summ(x):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,no_px=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan,top3_share=np.nan))
    p=pd.Series(pnl_no(x),index=x.index); m=p.groupby(x.month).sum()
    return pd.Series(dict(n=len(x),win=(~x.won).mean(),no_px=(1-x.price+SLIP).mean(),pnl=p.sum(),roi=p.sum()/(STAKE*len(x)),neg_m=(m<0).sum(),worst_m=m.min(),top3_share=p.nlargest(3).sum()/p.sum() if p.sum()>0 else np.nan))
def bymonth(x): p=pd.Series(pnl_no(x),index=x.index); return ' '.join(f"{m[-2:]}:{v:+.0f}" for m,v in p.groupby(x.month).sum().items())
band=lambda x: x.price.between(0.30,0.55)
dis=B[~B.agree]                                                   # disagreement nights
ridge_nights=dis[dis.modes.str.contains('ridge')]                 # Chicago, Houston, Dallas
rules={
 'A  mine: NO on FAVORITE, fav warmer than our YES, YES = disagree leg (ridge or ewma)': B[B.is_fav&B.ywhy.isin(['dis_ridge','dis_ewma'])&(B.lo>B.yes_lo)],
 'B  yours: NO on bucket 1 WARMER than the RIDGE pick, ridge-mode cities, disagreement nights': ridge_nights[ridge_nights.lo==ridge_nights.br+2],
 'B2 same as B but only if the ridge YES leg fired (ridge pick 5-60c)': ridge_nights[(ridge_nights.lo==ridge_nights.br+2)&(ridge_nights.ywhy=='dis_ridge')],
 'B3 1 warmer than the ridge pick, ALL 7 cities, disagreement nights (ignores city modes)': dis[dis.lo==dis.br+2],
 'B4 1 warmer than the ridge pick, all cities, AGREEMENT nights too': B[B.lo==B.br+2],
 'C  1 warmer than the city-mode model pick (ridge in ridge cities, ewma in ewma cities), disagreement nights': dis[(dis.modes.str.contains('ridge')&(dis.lo==dis.br+2))|(dis.modes.str.contains('ewma')&~dis.modes.str.contains('ridge')&(dis.lo==dis.be+2))],
 'D  1 warmer than the EWMA pick, ewma-mode cities (LA, Austin), disagreement nights': dis[dis.modes.str.contains('ewma')&(dis.lo==dis.be+2)],
 'E  A restricted to ridge-mode cities': B[B.is_fav&(B.ywhy=='dis_ridge')&(B.lo>B.yes_lo)],
 'F  B2 restricted to nights where that bucket is the favorite': ridge_nights[(ridge_nights.lo==ridge_nights.br+2)&(ridge_nights.ywhy=='dis_ridge')&ridge_nights.is_fav],
 'G  B2 where that bucket is NOT the favorite': ridge_nights[(ridge_nights.lo==ridge_nights.br+2)&(ridge_nights.ywhy=='dis_ridge')&~ridge_nights.is_fav],
 'H  union: A or B2': pd.concat([B[B.is_fav&B.ywhy.isin(['dis_ridge','dis_ewma'])&(B.lo>B.yes_lo)], ridge_nights[(ridge_nights.lo==ridge_nights.br+2)&(ridge_nights.ywhy=='dis_ridge')]]).drop_duplicates(['city','mday','lo']),
}
print("all rules: that bucket's YES price 0.30-0.55 (NO entry 0.46-0.71)\n")
R=pd.DataFrame({k:summ(v[band(v)]) for k,v in rules.items()}).T.round(2); print(R.to_string())
print("\nby month:")
for k in ['A  mine: NO on FAVORITE, fav warmer than our YES, YES = disagree leg (ridge or ewma)','B  yours: NO on bucket 1 WARMER than the RIDGE pick, ridge-mode cities, disagreement nights','B2 same as B but only if the ridge YES leg fired (ridge pick 5-60c)','B3 1 warmer than the ridge pick, ALL 7 cities, disagreement nights (ignores city modes)','C  1 warmer than the city-mode model pick (ridge in ridge cities, ewma in ewma cities), disagreement nights']:
    print(f"  {k[:3]} {bymonth(rules[k][band(rules[k])])}")
print("\nby city:")
for k in ['A  mine: NO on FAVORITE, fav warmer than our YES, YES = disagree leg (ridge or ewma)','B2 same as B but only if the ridge YES leg fired (ridge pick 5-60c)','B3 1 warmer than the ridge pick, ALL 7 cities, disagreement nights (ignores city modes)','C  1 warmer than the city-mode model pick (ridge in ridge cities, ewma in ewma cities), disagreement nights']:
    v=rules[k]; v=v[band(v)]; print(f"-- {k[:3]}"); print(v.groupby('city').apply(summ)[['n','win','no_px','pnl','roi','neg_m']].round(2).to_string())
a=rules['A  mine: NO on FAVORITE, fav warmer than our YES, YES = disagree leg (ridge or ewma)']; a=a[band(a)]; b=rules['B2 same as B but only if the ridge YES leg fired (ridge pick 5-60c)']; b=b[band(b)]
ka=set(zip(a.city,a.mday,a.lo)); kb=set(zip(b.city,b.mday,b.lo)); print(f"\noverlap: A={len(ka)} B2={len(kb)} both={len(ka&kb)} A-only={len(ka-kb)} B2-only={len(kb-ka)}")
print("A-only trades:", summ(a[[k not in kb for k in zip(a.city,a.mday,a.lo)]]).round(2).to_dict()); print("B2-only trades:", summ(b[[k not in ka for k in zip(b.city,b.mday,b.lo)]]).round(2).to_dict())
print("\nB2 by the ridge pick's own price (does a cheap ridge YES + NO on the warmer neighbour work?):"); yp=yk.price.to_dict(); b=b.assign(ypx=[yp.get((c,d)) for c,d in zip(b.city,b.mday)])
print(b.groupby(pd.cut(b.ypx,[0,.15,.25,.35,.45,.6]),observed=True).apply(summ)[['n','win','pnl','roi','neg_m']].round(2).to_string())
print("\nB price band sweep (ridge-mode cities, disagreement nights, 1 warmer than the ridge pick):"); v=rules['B  yours: NO on bucket 1 WARMER than the RIDGE pick, ridge-mode cities, disagreement nights']
print(pd.DataFrame({f"{lo:.2f}-{hi:.2f}":summ(v[v.price.between(lo,hi)]) for lo,hi in [(0.20,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.55),(0.55,0.65),(0.25,0.55),(0.30,0.55),(0.30,0.60),(0.35,0.55)]}).T[['n','win','no_px','pnl','roi','neg_m','worst_m']].round(2).to_string())
