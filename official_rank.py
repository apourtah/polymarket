"""Rank wallets by the OFFICIAL P&L (user-pnl-api curve, all markets) over Aug 7 - Sep 18, join taker-tape attribution and
closed-position volume; maker share = 1 - taker-bought $ / total bought $. -> out/official_rank.csv"""
import pandas as pd, numpy as np, json, glob, os
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_colwidth',28)
rows=[]
for f in glob.glob('data/pnl_curves/*.json'):
    w=os.path.basename(f)[:-5]
    try: p=pd.DataFrame(json.load(open(f)))
    except Exception: continue
    if len(p)==0 or 't' not in p: continue
    p['t']=pd.to_datetime(p.t,unit='s'); a=p[p.t<='2026-08-07 00:00']; b=p[p.t<='2026-09-19 00:00']
    if len(b)==0: continue
    start=a.p.iloc[-1] if len(a) else 0.0; rows.append(dict(w=w,pnl_6w=b.p.iloc[-1]-start,pnl_all=p.p.iloc[-1],since=str(p.t.min().date()),days_curve=len(p)))
O=pd.DataFrame(rows).set_index('w'); print(f"official curves: {len(O)} wallets")
Wr=pd.read_csv('out/real_winners.csv',index_col=0); Wall=pd.read_csv('out/wallets_all.csv',index_col=0); T=pd.read_parquet('out/wallet_fills.parquet')
tb=T[T.side=='BUY'].groupby('w').usd.sum(); O=O.join(Wr[['markets','days','cost','rpnl']],how='left').join(Wall[['fills','pnl','uname']].rename(columns={'pnl':'tape_pnl'}),how='left')
O['taker_buy_usd']=tb.reindex(O.index).fillna(0); O['maker_share']=(1-O.taker_buy_usd/O.cost).clip(0,1); O['roi_6w']=O.pnl_6w/O.cost
O.to_csv('out/official_rank.csv')
A=O[(O.markets>=10)&(O.days>=8)]
print(f"\nwallets >=10 weather markets & >=8 days: {len(A)}; official 6-week P&L: sum ${A.pnl_6w.sum():,.0f}, positive {int((A.pnl_6w>0).sum())}, > $1k: {int((A.pnl_6w>1000).sum())}, > $5k: {int((A.pnl_6w>5000).sum())}")
print("(taker-tape attribution for the same wallets sums to $%.0f; Spearman rank agreement %.3f)"%(A.tape_pnl.sum(),A[['pnl_6w','tape_pnl']].corr(method='spearman').iloc[0,1]))
print("\n### top 30 by OFFICIAL 6-week P&L")
print(A.sort_values('pnl_6w',ascending=False).head(30)[['pnl_6w','pnl_all','since','markets','days','cost','roi_6w','fills','tape_pnl','maker_share','uname']].round(2).to_string())
print("\n### official P&L by maker-share bucket (wallets >=10 markets): n, sum P&L, median ROI, share positive")
print(A.groupby(pd.cut(A.maker_share,[-0.01,0.2,0.5,0.8,1.0]),observed=True).agg(n=('pnl_6w','size'),pnl=('pnl_6w','sum'),med_roi=('roi_6w','median'),pos=('pnl_6w',lambda s:(s>0).mean()),cost=('cost','sum')).round(2).to_string())
top=A.pnl_6w.nlargest(30).index; print("\nshare of the top-30's P&L earned by wallets with maker_share >= 0.5: %.0f%%"%(100*A.loc[top][A.loc[top].maker_share>=0.5].pnl_6w.sum()/A.loc[top].pnl_6w.sum()))
print("of the top-30 by official P&L, how many were in our taker-tape top-30?", len(set(top)&set(Wall[(Wall.markets>=10)&(Wall.days>=8)].pnl.nlargest(30).index)))
print("\n### bottom 15 by official 6-week P&L"); print(A.sort_values('pnl_6w').head(15)[['pnl_6w','markets','cost','roi_6w','fills','tape_pnl','maker_share','uname']].round(2).to_string())
