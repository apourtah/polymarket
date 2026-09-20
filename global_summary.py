import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
STAKE=50; SLIP=0.01; MIN_PX=0.05
B=pd.read_parquet('out/global_buckets.parquet'); B['day']=pd.to_datetime(B.day); n=B.drop_duplicates(['city','day'])
print(f"{len(n)} city-nights, {n.city.nunique()} cities, {n.day.min().date()}..{n.day.max().date()}; agreement {n.agree.mean():.0%}; EWMA bucket hit {(n.be==n.actual).mean():.0%}, ridge hit {(n.br==n.actual).mean():.0%}, market favorite hit {(n.fav==n.actual).mean():.0%} (favorite avg price {n.fav_p.mean():.2f})")
B['month']=B.day.dt.to_period('M').astype(str); B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['is_fav']=B.lo==B.fav
def pnl(x,side):
    ask=np.where(side=='yes',x.price+SLIP,1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(side=='yes',x.won,~x.won); return np.where(win,sh-STAKE,-STAKE)-fee
def legs(X,modes=None):
    def ok(c,leg): return True if modes is None else leg in modes.get(c,set())
    ag=X[X.agree&X.is_be&X.price.between(0.10,0.53)&X.city.map(lambda c: ok(c,'agree'))].assign(leg='agree')
    dr=X[~X.agree&X.is_br&X.price.between(MIN_PX,0.45)&X.city.map(lambda c: ok(c,'ridge'))].assign(leg='ridge')
    de=X[~X.agree&X.is_be&X.price.between(MIN_PX,0.45)&X.city.map(lambda c: ok(c,'ewma'))].assign(leg='ewma')
    Y=pd.concat([ag,dr,de]).assign(side='yes'); Y=Y.sort_values('price').drop_duplicates(['city','day'])   # one YES per city-night (cheapest if both model legs qualify)
    yk=Y.set_index(['city','day']); X=X.assign(yes_lo=[yk.lo.get((c,d),np.nan) for c,d in zip(X.city,X.day)],yleg=[yk.leg.get((c,d)) for c,d in zip(X.city,X.day)])
    dis=X[X.yleg.isin(['ridge','ewma'])]; N=pd.concat([dis[dis.is_fav&(dis.lo>dis.yes_lo)],dis[(dis.yleg=='ridge')&(dis.lo==dis.br+1)]]).drop_duplicates(['city','day','lo'])
    N=N[N.price.between(0.35,0.55)].assign(leg='no_H',side='no'); T=pd.concat([Y,N]); T['pnl']=np.where(T.side=='yes',pnl(T,'yes'),pnl(T,'no')); return T
def summ(T):
    if len(T)==0: return pd.Series(dict(n=0,win=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    m=T.groupby('month').pnl.sum(); return pd.Series(dict(n=len(T),win=(T.pnl>0).mean(),pnl=T.pnl.sum(),roi=T.pnl.sum()/(STAKE*len(T)),neg_m=(m<0).sum(),worst_m=m.min()))
T=legs(B); print("\n### every leg in every city (no per-city modes), $50 clips, mids +1c, fee:"); print(pd.DataFrame({k:summ(T[T.leg==k]) for k in ['agree','ridge','ewma','no_H']}|{'ALL':summ(T)}).T.round(2).to_string())
print("\nby month:", T.groupby('month').pnl.sum().round(0).to_dict())
pc=T.groupby(['city','leg']).pnl.agg(['size','sum']); pc['roi']=pc['sum']/(STAKE*pc['size']); print("\nper city x leg (n, P&L, ROI):"); print(pc.round(2).unstack('leg').to_string())
print("\n### honest out-of-sample: choose each city's legs on Jun 15 - Aug 1 (keep legs with ROI>0 and n>=8), test Aug 2 - Sep 18")
tr=legs(B[B.day<='2026-08-01']); sel=tr.groupby(['city','leg']).pnl.agg(['size','sum']); modes={}
for (c,l),r in sel.iterrows():
    if l!='no_H' and r['size']>=8 and r['sum']>0: modes.setdefault(c,set()).add(l)
print("selected:", {c:sorted(v) for c,v in modes.items()})
te=legs(B[B.day>'2026-08-01'],modes); print(pd.DataFrame({k:summ(te[te.leg==k]) for k in ['agree','ridge','ewma','no_H']}|{'ALL':summ(te)}).T.round(2).to_string())
print("test by city:"); print(te.groupby('city').pnl.agg(['size','sum']).assign(roi=lambda d:d['sum']/(STAKE*d['size'])).round(2).sort_values('sum',ascending=False).to_string())
print("\nfor reference, same test window with all legs everywhere:", summ(legs(B[B.day>'2026-08-01'])).round(2).to_dict())
print("agree-only everywhere, whole period:", summ(T[T.leg=='agree']).round(2).to_dict(), "| test window:", summ(legs(B[B.day>'2026-08-01'])[lambda t:t.leg=='agree']).round(2).to_dict())
