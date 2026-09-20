"""Who really wins? Realised P&L from closed positions (includes maker fills and exits) vs our taker-tape attribution.
Weather highest-temperature markets, 7 cities, Aug 7 - Sep 18. Outputs out/real_winners.csv"""
import pandas as pd, numpy as np, json, glob, re, os
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_colwidth',30)
CITY=re.compile(r'highest temperature in (Los Angeles|Austin|Chicago|Houston|Dallas|Seattle|Miami) be'); FROM,TO='2026-08-07','2026-09-18'
rows=[]
for f in glob.glob('data/realized/*.json'):
    w=os.path.basename(f)[:-5]
    for x in json.load(open(f)):
        if x.get('title') and CITY.search(x['title']) and x.get('endDate') and FROM<=x['endDate']<=TO:
            rows.append(dict(w=w,cid=x['conditionId'],city=CITY.search(x['title']).group(1),day=x['endDate'],outcome=x['outcome'],bought=x['totalBought'] or 0,avg=x['avgPrice'] or 0,rpnl=x['realizedPnl'] or 0))
R=pd.DataFrame(rows); R['cost']=R.bought*R.avg; print(f"closed positions: {len(R):,} rows, {R.w.nunique():,} wallets, {R.cid.nunique():,} markets")
T=pd.read_parquet('out/wallet_fills.parquet'); tk=T.groupby('w').agg(tape_fills=('pnl','size'),tape_pnl=('pnl','sum'),tape_usd=('usd','sum'),tape_sh=('size','sum'))
Wr=R.groupby('w').agg(markets=('cid','nunique'),days=('day','nunique'),cost=('cost','sum'),bought_sh=('bought','sum'),rpnl=('rpnl','sum'),pos=('rpnl','size'))
dd=R.groupby(['w','day']).rpnl.sum().groupby('w').agg(['mean','std','count']); Wr['t']=dd['mean']/(dd['std']/np.sqrt(dd['count'])); Wr['day_win']=R.groupby(['w','day']).rpnl.sum().groupby('w').apply(lambda s:(s>0).mean())
Wr=Wr.join(tk,how='left').fillna({'tape_fills':0,'tape_pnl':0,'tape_usd':0,'tape_sh':0}); Wr['roi']=Wr.rpnl/Wr.cost; Wr['maker_share']=(1-Wr.tape_sh/Wr.bought_sh).clip(0,1)
Wall=pd.read_csv('out/wallets_all.csv',index_col=0); Wr['name']=[Wall.uname.get(w,'') if isinstance(Wall.uname.get(w,''),str) else '' for w in Wr.index]
Wr.to_csv('out/real_winners.csv')
print(f"\nrealised P&L sum over all wallets: ${Wr.rpnl.sum():,.0f} (tape attribution sum for the same wallets: ${Wr.tape_pnl.sum():,.0f})")
A=Wr[(Wr.markets>=10)&(Wr.days>=8)]
print("\n### top 25 by REALISED P&L (>=10 markets, >=8 days) — with taker-tape attribution and maker share")
print(A.sort_values('rpnl',ascending=False).head(25)[['markets','days','cost','rpnl','roi','day_win','t','tape_fills','tape_pnl','maker_share','name']].round(2).to_string())
print("\n### wallets whose realised P&L is far ABOVE what their taker fills explain (makers / good exits): rpnl - tape_pnl >= 1500")
print(A[(A.rpnl-A.tape_pnl)>=1500].sort_values('rpnl',ascending=False).head(20)[['markets','days','cost','rpnl','roi','day_win','t','tape_fills','tape_pnl','maker_share','name']].round(2).to_string())
print("\n### wallets whose realised P&L is far BELOW their taker attribution (scalpers who exit / hold-to-resolution overstates): tape_pnl - rpnl >= 1500")
print(A[(A.tape_pnl-A.rpnl)>=1500].sort_values('tape_pnl',ascending=False).head(15)[['markets','days','cost','rpnl','roi','tape_fills','tape_pnl','maker_share','name']].round(2).to_string())
print("\nrank agreement (Spearman) between realised P&L and tape attribution, wallets >=10 markets:", round(A[['rpnl','tape_pnl']].corr(method='spearman').iloc[0,1],3))
print("share of realised profit (positive wallets) earned by wallets with maker_share >= 0.5:", round(A[(A.rpnl>0)&(A.maker_share>=0.5)].rpnl.sum()/A[A.rpnl>0].rpnl.sum(),3))
print("\n### consistency: realised P&L split-half (odd vs even days) for wallets with >=10 markets")
days=sorted(R.day.unique()); odd=set(days[::2]); P1=R[R.day.isin(odd)].groupby('w').rpnl.sum(); P2=R[~R.day.isin(odd)].groupby('w').rpnl.sum(); c=pd.concat([P1,P2],axis=1).reindex(A.index).dropna(); c.columns=['a','b']
print(f"  Spearman {c.corr(method='spearman').iloc[0,1]:.3f}; top-20 on odd days -> {(c.loc[c.a.nlargest(20).index,'b']>0).sum()} positive on even days; bottom-20 -> {(c.loc[c.a.nsmallest(20).index,'b']<0).sum()} negative")
print("  top-20 realised winners on odd days and their even-day P&L:"); print(pd.DataFrame({'odd':c.a.nlargest(20).round(0),'even':c.b.reindex(c.a.nlargest(20).index).round(0),'name':Wr.name.reindex(c.a.nlargest(20).index),'maker':Wr.maker_share.reindex(c.a.nlargest(20).index).round(2)}).to_string())
