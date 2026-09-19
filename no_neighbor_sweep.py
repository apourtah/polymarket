"""NO on the buckets 1 / 2 away from our YES bucket (cooler, warmer, either), nights where our YES leg fired.
Sweep on that bucket's YES price (NO entry = 1 - price + 1c). out/backtest_evening_buckets.parquet, 7 cities, 21:35 mids."""
import pandas as pd, numpy as np, os
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
STAKE=50.0; SLIP=0.01
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B['month']=B.mday.dt.to_period('M').astype(str)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['modes']=B.city.map(lambda c: ','.join(sorted(MODES[c]))); B['is_fav']=B.lo==B.fav
AMIN,AMAX=float(os.environ.get('AMIN','0.10')),float(os.environ.get('AMAX','0.50'))
Y=pd.concat([B[B.agree&B.is_be&B.modes.str.contains('agree')&B.price.between(AMIN,AMAX)].assign(why='agree'),
             B[~B.agree&B.is_br&B.modes.str.contains('ridge')&B.price.between(0.05,0.60)].assign(why='dis_ridge'),
             B[~B.agree&B.is_be&B.modes.str.contains('ewma')&B.price.between(0.05,0.60)].assign(why='dis_ewma')])
yk=Y.set_index(['city','mday']); B['yes_lo']=[yk.lo.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['ywhy']=[yk.why.get((c,d)) for c,d in zip(B.city,B.mday)]
B['yes_px']=[yk.price.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]
N=B[B.yes_lo.notna()&(B.lo>-999)&(B.hi<999)].copy(); N['off']=((N.lo-N.yes_lo)/2).round().astype(int)   # + = warmer than our YES bucket
def pnl_no(x): ask=(1-x.price+SLIP).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; return np.where(~x.won,sh-STAKE,-STAKE)-fee
def summ(x):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,no_px=np.nan,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    p=pd.Series(pnl_no(x),index=x.index); m=p.groupby(x.month).sum()
    return pd.Series(dict(n=len(x),win=(~x.won).mean(),no_px=(1-x.price+SLIP).mean(),pnl=p.sum(),roi=p.sum()/(STAKE*len(x)),neg_m=(m<0).sum(),worst_m=m.min()))
def bymonth(x): p=pd.Series(pnl_no(x),index=x.index); return ' '.join(f"{m[-2:]}:{v:+.0f}" for m,v in p.groupby(x.month).sum().items())
print(f"YES leg: agree {AMIN}-{AMAX} / model 0.05-0.60 -> {len(Y)} city-nights with a YES\n")
sets={'1 cooler':N[N.off==-1],'1 warmer':N[N.off==1],'1 either':N[N.off.abs()==1],'2 cooler':N[N.off==-2],'2 warmer':N[N.off==2],'2 either':N[N.off.abs()==2],'1+2 cooler':N[N.off.isin([-1,-2])],'1+2 warmer':N[N.off.isin([1,2])],'1+2 either':N[N.off.abs().isin([1,2])]}
print("### A. unconditional on price: NO on the neighbour bucket(s)")
print(pd.DataFrame({k:summ(v) for k,v in sets.items()}).T.round(2).to_string())
print("\n### B. sweep on the neighbour's YES price band (NO entry = 1 - price + 1c)")
bands=[(0.0,0.10),(0.10,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),(0.50,0.60),(0.60,1.0)]
for k in ['1 cooler','1 warmer','1 either','2 cooler','2 warmer','2 either']:
    v=sets[k]; print(f"\n-- {k}: price distribution", v.price.describe()[['25%','50%','75%']].round(2).to_dict())
    print(pd.DataFrame({f"yes {a:.2f}-{b:.2f} (NO {1-b+.01:.2f}-{1-a+.01:.2f})":summ(v[v.price.between(a,b)]) for a,b in bands}).T.round(2).to_string())
print("\n### C. cumulative bands x offset -> ROI (n)")
cum=[(0.20,0.60),(0.25,0.55),(0.30,0.55),(0.30,0.60),(0.35,0.60),(0.40,0.70)]
print(pd.DataFrame({k:{f"{a:.2f}-{b:.2f}":f"{summ(v[v.price.between(a,b)]).roi:+.2f} ({len(v[v.price.between(a,b)])})" for a,b in cum} for k,v in sets.items()}).T.to_string())
print("\n### D. is the neighbour the market favorite? (1 either, yes 0.30-0.60)")
v=sets['1 either']; v=v[v.price.between(0.30,0.60)]; print(pd.DataFrame({'neighbour IS the favorite':summ(v[v.is_fav]),'neighbour is not the favorite':summ(v[~v.is_fav])}).T.round(2).to_string())
print("\n### E. by our YES leg type (1 warmer / 1 cooler, yes 0.30-0.60)")
for k in ['1 warmer','1 cooler']:
    v=sets[k]; v=v[v.price.between(0.30,0.60)]; print(f"-- {k}:"); print(v.groupby('ywhy').apply(summ).round(2).to_string())
print("\n### F. by city / month for the best cells")
for k,(a,b) in [('1 warmer',(0.30,0.60)),('1 either',(0.30,0.60)),('1+2 warmer',(0.30,0.60))]:
    v=sets[k]; v=v[v.price.between(a,b)]; print(f"\n-- {k}, yes {a:.2f}-{b:.2f}: {summ(v).round(2).to_dict()}"); print(v.groupby('city').apply(summ).round(2).to_string()); print("by month:", bymonth(v))
print("\n### G. both neighbours at once (1 cooler AND 1 warmer, each yes 0.30-0.60) vs only the warmer / only the cooler on the same nights")
w=sets['1 warmer']; c=sets['1 cooler']; w=w[w.price.between(0.3,0.6)]; c=c[c.price.between(0.3,0.6)]
both=set(zip(w.city,w.mday))&set(zip(c.city,c.mday)); print(f"nights where both neighbours are 30-60c: {len(both)}")
print(pd.DataFrame({'warmer on those nights':summ(w[[k in both for k in zip(w.city,w.mday)]]),'cooler on those nights':summ(c[[k in both for k in zip(c.city,c.mday)]])}).T.round(2).to_string())
