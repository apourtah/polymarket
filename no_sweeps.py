"""Sweeps for the NO legs on out/backtest_evening_buckets.parquet (7 production cities, 21:35 local eve-before mids).
 4) yes/yes: NO on the favorite on nights our YES leg fired -> sweep the favorite's price / NO entry price, and the condition.
 5) no/no: NO on the runner-up -> sweep favorite price x runner-up price x third price / gaps."""
import pandas as pd, numpy as np, os
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
STAKE=50.0; SLIP=0.01
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B['is_fav']=B.lo==B.fav; B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['modes']=B.city.map(lambda c: ','.join(sorted(MODES[c])))
B['rank_px']=B.groupby(['city','mday']).price.rank(ascending=False,method='first'); B['month']=B.mday.dt.to_period('M').astype(str)
AMIN,AMAX=float(os.environ.get('AMIN','0.10')),float(os.environ.get('AMAX','0.50'))
Y=pd.concat([B[B.agree&B.is_be&B.modes.str.contains('agree')&B.price.between(AMIN,AMAX)].assign(why='agree'),
             B[~B.agree&B.is_br&B.modes.str.contains('ridge')&B.price.between(0.05,0.60)].assign(why='dis_ridge'),
             B[~B.agree&B.is_be&B.modes.str.contains('ewma')&B.price.between(0.05,0.60)].assign(why='dis_ewma')])
yes_nights=set(zip(Y.city,Y.mday)); B['yes_fired']=[(c,d) in yes_nights for c,d in zip(B.city,B.mday)]
ylo=Y.set_index(['city','mday']).lo.to_dict(); B['yes_lo']=[ylo.get((c,d)) for c,d in zip(B.city,B.mday)]
def pnl_no(x):
    ask=(1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(~x.won,sh-STAKE,-STAKE)-fee
def summ(x):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,no_px=np.nan,stake=0,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    p=pd.Series(pnl_no(x),index=x.index); m=p.groupby(x.month).sum()
    return pd.Series(dict(n=len(x),win=(~x.won).mean(),no_px=(1-x.price+SLIP).mean(),stake=STAKE*len(x),pnl=p.sum(),roi=p.sum()/(STAKE*len(x)),neg_m=(m<0).sum(),worst_m=m.min()))
print(f"YES leg (agree {AMIN}-{AMAX}, model 0.05-0.60): {len(Y)} legs on {len(yes_nights)} city-nights\n")
print("#### 4. NO on the FAVORITE, nights where our YES fired (favorite != our YES bucket)")
F=B[B.is_fav&B.yes_fired&(B.lo!=B.yes_lo)].copy(); F['fav_off_both']=(~F.is_be)&(~F.is_br)
print("condition on the favorite vs our models:"); print(pd.DataFrame({'fav is the OTHER model bucket':summ(F[~F.fav_off_both]),'fav is neither model bucket':summ(F[F.fav_off_both]),'any (fav != our YES)':summ(F)}).T.round(2).to_string())
print("\nsweep: favorite YES price band (NO entry = 1 - price + 1c), fav != our YES, any model condition:")
bands=[(0.20,0.30),(0.30,0.35),(0.35,0.40),(0.40,0.45),(0.45,0.50),(0.50,0.55),(0.55,0.60),(0.60,0.70),(0.70,0.85)]
print(pd.DataFrame({f"fav {a:.2f}-{b:.2f} (NO {1-b+.01:.2f}-{1-a+.01:.2f})":summ(F[F.price.between(a,b)]) for a,b in bands}).T.round(2).to_string())
print("\nsame, fav neither model bucket:")
print(pd.DataFrame({f"fav {a:.2f}-{b:.2f}":summ(F[F.fav_off_both&F.price.between(a,b)]) for a,b in bands}).T.round(2).to_string())
print("\ncumulative bands (fav != our YES, any model condition):")
print(pd.DataFrame({f"fav {a:.2f}-{b:.2f}":summ(F[F.price.between(a,b)]) for a,b in [(0.30,0.60),(0.30,0.50),(0.35,0.60),(0.35,0.55),(0.40,0.60),(0.30,0.70),(0.20,0.60)]}).T.round(2).to_string())
print("\nsplit by our YES leg type x fav band 0.30-0.60:"); F2=F[F.price.between(0.30,0.60)]; wy=Y.set_index(['city','mday']).why.to_dict(); F2=F2.assign(ywhy=[wy.get((c,d)) for c,d in zip(F2.city,F2.mday)])
print(F2.groupby('ywhy').apply(summ).round(2).to_string())
print("\nby city, fav 0.30-0.60, any model condition:"); print(F2.groupby('city').apply(summ).round(2).to_string())
print("\nby city, fav 0.30-0.60, fav neither model bucket:"); print(F2[F2.fav_off_both].groupby('city').apply(summ).round(2).to_string())
print("\ndistance favorite - our YES bucket (buckets), fav 0.30-0.60:"); F2=F2.assign(dist=((F2.lo-F2.yes_lo)/2).clip(-3,3)); print(F2.groupby('dist').apply(summ).round(2).to_string())
print("\nour YES price x fav band (does a cheap YES + NO fav pair work?):"); yp=Y.set_index(['city','mday']).price.to_dict(); F2=F2.assign(ypx=[yp.get((c,d)) for c,d in zip(F2.city,F2.mday)])
print(F2.groupby(pd.cut(F2.ypx,[0,.15,.25,.35,.45,.6]),observed=True).apply(summ).round(2).to_string())

print("\n\n#### 5. NO on the RUNNER-UP (2nd highest priced bucket): sweep favorite price x runner-up price x third price")
R=B[B.rank_px==2].copy(); T=B[B.rank_px==3][['city','mday','price']].rename(columns={'price':'third_p'}); R=R.merge(T,on=['city','mday'],how='left')
R['gap12']=R.fav_p-R.price; R['gap23']=R.price-R.third_p
print("all nights: favorite band x runner-up band -> ROI (n)"); 
fb=pd.cut(R.fav_p,[0,.3,.4,.5,.6,.7,1]); rb=pd.cut(R.price,[0,.15,.2,.25,.3,.35,.4,.5])
tab=R.groupby([fb,rb],observed=True).apply(lambda x: f"{summ(x).roi:+.2f} ({len(x)})").unstack(); print(tab.to_string())
print("\nrunner-up NO win rate (bucket loses) by the same grid:"); print(R.groupby([fb,rb],observed=True).apply(lambda x: f"{(~x.won).mean():.2f}").unstack().to_string())
print("\nrunner-up band x third-bucket band -> ROI (n):"); tb=pd.cut(R.third_p,[0,.05,.1,.15,.2,.3,.5])
print(R.groupby([rb,tb],observed=True).apply(lambda x: f"{summ(x).roi:+.2f} ({len(x)})").unstack().to_string())
print("\ngap favorite-runner-up x gap runner-up-third -> ROI (n):"); g1=pd.cut(R.gap12,[-1,0.05,0.1,0.2,0.3,1]); g2=pd.cut(R.gap23,[-1,0.05,0.1,0.15,0.2,0.3,1])
print(R.groupby([g1,g2],observed=True).apply(lambda x: f"{summ(x).roi:+.2f} ({len(x)})").unstack().to_string())
print("\nrunner-up warmer vs cooler than favorite, by runner-up band:"); R['side']=np.where(R.lo>R.fav,'warmer','cooler')
print(R.groupby(['side',rb],observed=True).apply(summ).round(2).to_string())
print("\nrunner-up vs our models: is the runner-up a model bucket? by runner-up band"); R['is_model']=R.is_be|R.is_br
print(R.groupby(['is_model',rb],observed=True).apply(summ).round(2).to_string())
print("\nrunner-up NO on nights with / without our YES, by runner-up band:")
print(R.groupby(['yes_fired',rb],observed=True).apply(summ).round(2).to_string())
print("\nbest-looking cells re-checked by month (runner-up not a model bucket, runner-up 0.20-0.35, favorite 0.40-0.70):")
X=R[~R.is_model&R.price.between(0.20,0.35)&R.fav_p.between(0.40,0.70)]; print(summ(X).round(2).to_dict())

print("\n\n#### 4b. NO-on-favorite, refined: which pairing works (fav != our YES)")
wy=Y.set_index(['city','mday']).why.to_dict(); F=F.assign(ywhy=[wy.get((c,d)) for c,d in zip(F.city,F.mday)], dist=(F.lo-F.yes_lo)/2)
conds={'any YES':F, 'YES = model-disagree leg (ridge/ewma)':F[F.ywhy!='agree'], 'YES = agree leg':F[F.ywhy=='agree'],
       'fav neither model bucket':F[F.fav_off_both], 'fav neither & YES = disagree leg':F[F.fav_off_both&(F.ywhy!='agree')],
       'fav warmer than our YES':F[F.dist>0], 'fav warmer & YES = disagree leg':F[(F.dist>0)&(F.ywhy!='agree')],
       'fav cooler than our YES':F[F.dist<0]}
for a,b in [(0.30,0.50),(0.35,0.50),(0.30,0.55),(0.35,0.55),(0.30,0.60)]:
    print(f"\nfavorite YES price {a:.2f}-{b:.2f}  (NO entry {1-b+.01:.2f}-{1-a+.01:.2f}):")
    print(pd.DataFrame({k:summ(v[v.price.between(a,b)]) for k,v in conds.items()}).T.round(2).to_string())
print("\nfav 0.30-0.55, YES = disagree leg, by city:"); Z=F[(F.ywhy!='agree')&F.price.between(0.30,0.55)]; print(Z.groupby('city').apply(summ).round(2).to_string())
def bymonth(Z): p=pd.Series(pnl_no(Z),index=Z.index); return ' '.join(f"{m[-2:]}:{v:+.0f}({n})" for (m,v),n in zip(p.groupby(Z.month).sum().items(),Z.groupby('month').size()))
print("by month:", bymonth(Z))
W=F[(F.ywhy!='agree')&(F.dist>0)&F.price.between(0.30,0.55)]; print("\nfav WARMER than our YES & YES = disagree leg, fav 0.30-0.55: by city"); print(W.groupby('city').apply(summ).round(2).to_string()); print("by month:", bymonth(W))
W2=F[(F.dist>0)&F.price.between(0.30,0.55)]; print("\nfav WARMER than our YES, any YES leg, fav 0.30-0.55: by city"); print(W2.groupby('city').apply(summ).round(2).to_string()); print("by month:", bymonth(W2))
print("\nfav 0.30-0.55, any YES, by city:"); Z=F[F.price.between(0.30,0.55)]; print(Z.groupby('city').apply(summ).round(2).to_string())
