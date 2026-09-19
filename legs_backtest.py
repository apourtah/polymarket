"""Leg-by-leg backtest on out/backtest_evening_buckets.parquet (every priced bucket per production city-night at 21:35
local the evening before, with the walk-forward EWMA/ridge P per bucket). $50 per leg, +1c slip, taker fee.
Legs: YES (production per-city rule, band variants), extra YES (2nd bucket), NO on the favorite (yes/yes), NO on the
runner-up (no/no), NO on any bucket both models call overpriced.  env: STAKE, SLIP, START, END"""
import pandas as pd, numpy as np, os, sys
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
STAKE=float(os.environ.get('STAKE','50')); SLIP=float(os.environ.get('SLIP','0.01'))
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday)
B=B[(B.mday>=os.environ.get('START','2026-01-18'))&(B.mday<=os.environ.get('END','2026-09-16'))].copy()
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B['pav']=(B.pe+B.pr)/2; B['is_fav']=B.lo==B.fav; B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br
B['modes']=B.city.map(lambda c: ','.join(sorted(MODES.get(c,set()))))
B['rank_pav']=B.groupby(['city','mday']).pav.rank(ascending=False,method='first')
B['rank_px']=B.groupby(['city','mday']).price.rank(ascending=False,method='first')
B['fav_off_models']=(~B.groupby(['city','mday']).apply(lambda g: pd.Series((g.fav==g.be)|(g.fav==g.br),index=g.index)).reset_index(level=[0,1],drop=True))  # favorite != both model buckets
B['month']=B.mday.dt.to_period('M').astype(str)
def pnl_yes(x):
    ask=(x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(x.won,sh-STAKE,-STAKE)-fee
def pnl_no(x):
    ask=(1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(~x.won,sh-STAKE,-STAKE)-fee
def summ(x,side):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,px=np.nan,stake=0,pnl=0,roi=np.nan))
    p=pnl_yes(x) if side=='yes' else pnl_no(x); w=x.won if side=='yes' else ~x.won; px=x.price if side=='yes' else 1-x.price
    return pd.Series(dict(n=len(x),win=w.mean(),px=px.mean(),stake=STAKE*len(x),pnl=p.sum(),roi=p.sum()/(STAKE*len(x))))
def yes_leg(amin,amax,mmin,mmax):
    ag=B[B.agree&B.is_be&B.modes.str.contains('agree')&B.price.between(amin,amax)].assign(why='agree')
    dr=B[~B.agree&B.is_br&B.modes.str.contains('ridge')&B.price.between(mmin,mmax)].assign(why='dis_ridge')
    de=B[~B.agree&B.is_be&B.modes.str.contains('ewma')&B.price.between(mmin,mmax)].assign(why='dis_ewma')
    return pd.concat([ag,dr,de])
print(f"=== {B.groupby(['city','mday']).ngroups} city-nights, {B.mday.min().date()}..{B.mday.max().date()}, ${STAKE:.0f}/leg, slip {SLIP}\n")
print("### 1. YES band. Production = agree 0.35-0.60, model buckets 0.05-0.60. Per-leg P&L by entry price bin (unrestricted bands):")
Y=yes_leg(0.0,1.0,0.0,1.0); Y['bin']=pd.cut(Y.price,[0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,1])
print(Y.groupby(['why','bin'],observed=True).apply(lambda x: summ(x,'yes')).round(2).to_string())
print("\nband variants (agree_min-agree_max / model_min-model_max):")
rows=[]
for am,ax,mm,mx in [(0.35,0.60,0.05,0.60),(0.35,0.50,0.05,0.60),(0.35,0.45,0.05,0.60),(0.30,0.50,0.05,0.60),(0.25,0.50,0.05,0.60),(0.20,0.50,0.05,0.60),(0.35,0.60,0.05,0.40),(0.35,0.50,0.05,0.35),(0.35,0.50,0.05,0.30),(0.35,0.50,0.10,0.35),(0.35,0.50,0.08,0.40),(0.30,0.50,0.08,0.40)]:
    y=yes_leg(am,ax,mm,mx); s=summ(y,'yes'); s['agree_roi']=summ(y[y.why=='agree'],'yes').roi; s['model_roi']=summ(y[y.why!='agree'],'yes').roi
    p=pd.Series(pnl_yes(y),index=y.index); m=y.assign(p=p).groupby('month').p.sum(); s['neg_months']=(m<0).sum(); s['worst_m']=m.min()
    rows.append(pd.Series(s,name=f"{am:.2f}-{ax:.2f} / {mm:.2f}-{mx:.2f}"))
print(pd.DataFrame(rows).round(2).to_string())

print("\n### 2. Extra YES positions per city (on top of the production YES):")
prod=yes_leg(0.35,0.60,0.05,0.60); pk=set(zip(prod.city,prod.mday,prod.lo)); nights_traded=set(zip(prod.city,prod.mday))
B['in_prod']=[(c,d,l) in pk for c,d,l in zip(B.city,B.mday,B.lo)]; B['night_traded']=[(c,d) in nights_traded for c,d in zip(B.city,B.mday)]
cands={
 'other model bucket on disagree nights (not the city mode), <=0.40': B[~B.agree&(B.is_be|B.is_br)&~B.in_prod&(B.price<=0.40)],
 '2nd-best avg-P bucket, <=0.25': B[(B.rank_pav==2)&~B.in_prod&(B.price<=0.25)],
 '2nd-best avg-P bucket, 0.08-0.25': B[(B.rank_pav==2)&~B.in_prod&B.price.between(0.08,0.25)],
 '2nd-best avg-P bucket, 0.08-0.25, avgP-price>=0.05': B[(B.rank_pav==2)&~B.in_prod&B.price.between(0.08,0.25)&(B.pav-B.price>=0.05)],
 'bucket 1 cooler than production bucket, 0.08-0.25': B[np.array([(c,d,l+2) in pk for c,d,l in zip(B.city,B.mday,B.lo)])&B.price.between(0.08,0.25)],
 'bucket 1 warmer than production bucket, 0.08-0.25': B[np.array([(c,d,l-2) in pk for c,d,l in zip(B.city,B.mday,B.lo)])&B.price.between(0.08,0.25)],
 'bucket 1 cooler than FAVORITE (their tilt), 0.08-0.25, not in prod': B[(B.lo==B.fav-2)&~B.in_prod&B.price.between(0.08,0.25)],
 'any bucket with avgP - price >= 0.10, price 0.08-0.30, not in prod': B[~B.in_prod&B.price.between(0.08,0.30)&(B.pav-B.price>=0.10)],
 'any bucket with min(Pe,Pr) - price >= 0.08, price 0.08-0.30, not in prod': B[~B.in_prod&B.price.between(0.08,0.30)&(B[['pe','pr']].min(axis=1)-B.price>=0.08)],
}
print(pd.DataFrame({k:summ(v,'yes') for k,v in cands.items()}).T.round(2).to_string())

print("\n### 3a. yes/yes: NO on the favorite when it is not either model's bucket, fav price 0.30-0.60")
F=B[B.is_fav&B.fav_off_models&B.price.between(0.30,0.60)]
print(pd.DataFrame({'all such nights':summ(F,'no'),'only nights where production YES fired':summ(F[F.night_traded],'no'),'nights with no production YES':summ(F[~F.night_traded],'no')}).T.round(2).to_string())
print("by fav price band (all such nights):"); print(F.groupby(pd.cut(F.price,[.3,.4,.5,.6]),observed=True).apply(lambda x: summ(x,'no')).round(2).to_string())
print("by city:"); print(F.groupby('city').apply(lambda x: summ(x,'no')).round(2).to_string())
print("by month:"); print(F.groupby('month').apply(lambda x: summ(x,'no')).round(2).to_string())

print("\n### 3b. no/no: NO on the runner-up (2nd-highest priced bucket), various conditions")
R=B[B.rank_px==2]
conds={
 'runner-up, any night, YES px 0.20-0.45': R[R.price.between(0.20,0.45)],
 'runner-up warmer than favorite, YES px 0.20-0.45': R[(R.lo>R.fav)&R.price.between(0.20,0.45)],
 'runner-up cooler than favorite, YES px 0.20-0.45': R[(R.lo<R.fav)&R.price.between(0.20,0.45)],
 'runner-up not a model bucket, px 0.20-0.45': R[~R.is_be&~R.is_br&R.price.between(0.20,0.45)],
 'runner-up not a model bucket, px 0.20-0.45, both models P<=0.15': R[~R.is_be&~R.is_br&R.price.between(0.20,0.45)&(R[['pe','pr']].max(axis=1)<=0.15)],
 'runner-up, fav <=0.60, not model bucket, px 0.20-0.45': R[(R.fav_p<=0.60)&~R.is_be&~R.is_br&R.price.between(0.20,0.45)],
 'runner-up, nights with NO production YES, px 0.20-0.45': R[~R.night_traded&R.price.between(0.20,0.45)],
 'runner-up, nights with NO production YES, not model bucket': R[~R.night_traded&~R.is_be&~R.is_br&R.price.between(0.20,0.45)],
}
print(pd.DataFrame({k:summ(v,'no') for k,v in conds.items()}).T.round(2).to_string())

print("\n### 3c. model-driven NO: any bucket where BOTH models say it is overpriced (price - max(Pe,Pr) >= edge), YES price >= 0.15")
rows=[]
for edge in [0.05,0.10,0.15,0.20]:
    for lo_px in [0.15,0.25,0.35]:
        X=B[(B.price>=lo_px)&(B.price-B[['pe','pr']].max(axis=1)>=edge)]; s=summ(X,'no'); s['on_fav_share']=X.is_fav.mean() if len(X) else np.nan
        p=pd.Series(pnl_no(X),index=X.index); m=X.assign(p=p).groupby('month').p.sum(); s['neg_months']=(m<0).sum(); s['worst_m']=m.min() if len(m) else np.nan
        rows.append(pd.Series(s,name=f"edge>={edge:.2f}, px>={lo_px:.2f}"))
print(pd.DataFrame(rows).round(2).to_string())
X=B[(B.price>=0.25)&(B.price-B[['pe','pr']].max(axis=1)>=0.10)]; print("\n  edge>=0.10, px>=0.25 by city:"); print(X.groupby('city').apply(lambda x: summ(x,'no')).round(2).to_string())
print("  ...by month:"); print(X.groupby('month').apply(lambda x: summ(x,'no')).round(2).to_string())
print("  ...favorite vs non-favorite:"); print(X.groupby('is_fav').apply(lambda x: summ(x,'no')).round(2).to_string())

print("\n\n######## DEEP DIVE ########")
O=B[~B.agree&(B.is_be|B.is_br)&~B.in_prod&(B.price<=0.40)].copy(); O['which']=np.where(O.is_be,'ewma','ridge')
print("### other-model bucket on disagree nights, <=0.40: by city x which model")
print(O.groupby(['city','which']).apply(lambda x: summ(x,'yes')).round(2).to_string())
print("by price bin:"); print(O.groupby(pd.cut(O.price,[0,.05,.1,.15,.2,.25,.3,.35,.4]),observed=True).apply(lambda x: summ(x,'yes')).round(2).to_string())
print("by month:"); print(O.groupby('month').apply(lambda x: summ(x,'yes')).round(2).to_string())
O['p']=pnl_yes(O); print("top 6 wins:"); print(O.sort_values('p',ascending=False).head(6)[['city','mday','lo','price','pe','pr','p']].round(3).to_string()); print(f"without top 3: pnl {O.p.sum()-O.p.nlargest(3).sum():.0f} on {STAKE*len(O):.0f}")
print("\n### agree leg 0.20-0.35 by month (is the cheap agree band stable?)")
A=B[B.agree&B.is_be&B.modes.str.contains('agree')]
for lo,hi in [(0.20,0.35),(0.30,0.35),(0.35,0.50),(0.50,0.60)]:
    x=A[A.price.between(lo,hi)].copy(); x['p']=pnl_yes(x); print(f"  {lo:.2f}-{hi:.2f}: n={len(x)} roi={x.p.sum()/(STAKE*len(x)):+.2f}  months: "+' '.join(f"{m[-2:]}:{v:+.0f}" for m,v in x.groupby('month').p.sum().items()))
print("\n### yes/yes NO-on-favorite, only nights where production YES fired: by city / month")
F2=F[F.night_traded]; print(F2.groupby('city').apply(lambda x: summ(x,'no')).round(2).to_string()); print(F2.groupby('month').apply(lambda x: summ(x,'no')).round(2).to_string())
print("by fav price band:"); print(F2.groupby(pd.cut(F2.price,[.3,.4,.5,.6]),observed=True).apply(lambda x: summ(x,'no')).round(2).to_string())
print("\n### PORTFOLIOS ($50 per leg): production YES vs proposed")
def portfolio(yes_df, extra_yes=None, no_fav=False):
    legs=[yes_df.assign(side='yes',leg='yes')]
    if extra_yes is not None: legs.append(extra_yes.assign(side='yes',leg='yes2'))
    if no_fav:
        nt=set(zip(yes_df.city,yes_df.mday)); f=B[B.is_fav&B.fav_off_models&B.price.between(0.30,0.60)]; f=f[[(c,d) in nt for c,d in zip(f.city,f.mday)]]
        legs.append(f.assign(side='no',leg='no_fav'))
    L=pd.concat(legs); L['p']=np.where(L.side=='yes',pnl_yes(L),pnl_no(L))
    m=L.groupby('month').p.sum(); daily=L.groupby('mday').p.sum().sort_index(); eq=daily.cumsum(); dd=(eq-eq.cummax()).min()
    nights=L.groupby(['city','mday']).size()
    return pd.Series(dict(legs=len(L),stake=STAKE*len(L),pnl=L.p.sum(),roi=L.p.sum()/(STAKE*len(L)),neg_months=(m<0).sum(),worst_m=m.min(),max_dd=dd,legs_per_night=len(L)/L.mday.nunique(),pct_nights_multi=(nights>1).mean(),
                          **{f"pnl_{k}":v for k,v in L.groupby('leg').p.sum().items()}))
Y0=yes_leg(0.35,0.60,0.05,0.60); Y1=yes_leg(0.30,0.50,0.05,0.60); Y2=yes_leg(0.30,0.50,0.08,0.40)
def other(y,mx=0.40):
    k=set(zip(y.city,y.mday,y.lo)); return B[~B.agree&(B.is_be|B.is_br)&~np.array([(c,d,l) in k for c,d,l in zip(B.city,B.mday,B.lo)])&(B.price<=mx)&(B.price>=0.05)]
R=pd.DataFrame({
 'A production (0.35-0.60 / 0.05-0.60)':portfolio(Y0),
 'B band 0.30-0.50 / 0.05-0.60':portfolio(Y1),
 'C band 0.30-0.50 / 0.08-0.40':portfolio(Y2),
 'D = B + other model bucket <=0.40':portfolio(Y1,other(Y1)),
 'E = B + NO fav (yes/yes)':portfolio(Y1,None,True),
 'F = B + other model + NO fav':portfolio(Y1,other(Y1),True),
 'G = C + other model <=0.40 + NO fav':portfolio(Y2,other(Y2),True),
}).T
print(R.round(2).to_string())

print("\n### NO-fav restricted to cities where it works (Dallas, Houston, LA, Austin, Seattle)")
def portfolio2(yes_df, cities):
    nt=set(zip(yes_df.city,yes_df.mday)); f=B[B.is_fav&B.fav_off_models&B.price.between(0.30,0.60)&B.city.isin(cities)]; f=f[[(c,d) in nt for c,d in zip(f.city,f.mday)]]
    L=pd.concat([yes_df.assign(side='yes',leg='yes'),f.assign(side='no',leg='no_fav')]); L['p']=np.where(L.side=='yes',pnl_yes(L),pnl_no(L))
    m=L.groupby('month').p.sum(); daily=L.groupby('mday').p.sum().sort_index(); eq=daily.cumsum()
    print(pd.Series(dict(legs=len(L),stake=STAKE*len(L),pnl=L.p.sum(),roi=L.p.sum()/(STAKE*len(L)),pnl_no_fav=f.assign(p=pnl_no(f)).p.sum(),no_fav_n=len(f),neg_months=(m<0).sum(),worst_m=m.min(),max_dd=(eq-eq.cummax()).min())).round(2).to_string())
    print("  by month:", ' '.join(f"{k[-2:]}:{v:+.0f}" for k,v in m.items()))
    print("  positions per traded city-night:", L.groupby(['city','mday']).size().value_counts().sort_index().to_dict())
portfolio2(Y1,['Dallas','Houston','Los Angeles','Austin','Seattle'])
