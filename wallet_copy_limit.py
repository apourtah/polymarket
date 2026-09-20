"""Copy-trade with a LIMIT order at the copied trader's own fill price, resting until resolution.
Fill model from the taker tape: our long-YES bid at p is filled when a later taker print sells YES to the book at p_yes <= p
(SELL Yes or BUY No); our NO bid (short YES at p) is filled when a later taker print buys YES / sells NO at p_yes >= p.
'strict' = price strictly better than ours (certain fill), 'incl' = also equal price (queue priority needed).
Reports fill rate, time-to-fill, our P&L (their price, filled orders only) vs their P&L on the same fills. $10 per order."""
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
T=pd.read_parquet('out/wallet_fills.parquet',columns=['w','cid','day','ts','side','outcome','p_yes','q','usd','pnl','won']).sort_values('ts'); W=pd.read_csv('out/wallets_all.csv',index_col=0); O=pd.read_csv('out/official_rank.csv',index_col=0)
TRAIN_END='2026-08-27'; tr=T[T.day<=TRAIN_END]; te=T[T.day>TRAIN_END]
Wtr=tr.groupby('w').agg(pnl=('pnl','sum'),mk=('cid','nunique'),days=('day','nunique')); dd=tr.groupby(['w','day']).pnl.sum().groupby('w').agg(['mean','std','count']); Wtr['t']=dd['mean']/(dd['std']/np.sqrt(dd['count']))
ALPHA=Wtr[(Wtr.mk>=8)&(Wtr.days>=6)&(Wtr.pnl>300)&(Wtr.t>1.0)].sort_values('pnl',ascending=False).head(15).index.tolist()
GROUPB=['0x9506e646497107cabf2d5b941a8e6a60d0db1c4f','0x331bf91c132af9d921e1908ca0979363fc47193f','0x180e62e6f035dbf69118a2306df25d28762129df','0xdd22fb0e71982d6793e9a22e47306c462eba2013','0xb6fbce093cdd139858c44148a6598d8ec028c038','0x56b381017b589fb0afb049137c8d4af2e7233fa2','0x95d381e71dba6f1c3199fd9bc040383d6fae6eff','0xbf13934a1fec7d3211fc15c138d84ac2a691b91a']
SET=list(dict.fromkeys(ALPHA+GROUPB)); name=lambda w: (W.uname.get(w,'') if isinstance(W.uname.get(w,''),str) and not str(W.uname.get(w,'')).startswith('0x') else w[:8])
# index the tape per market: arrays of ts, p_yes, taker-sells-YES flag
idx={}
for cid,g in te.groupby('cid'):
    sells_yes=((g.side=='SELL')&(g.outcome=='Yes'))|((g.side=='BUY')&(g.outcome=='No')); idx[cid]=(g.ts.values,g.p_yes.values,sells_yes.values,g.w.values)
STAKE=10.0
def fill(row,strict):
    ts,py,sy,ww=idx[row.cid]; m=(ts>row.ts)&(ww!=row.w)
    if row.q>0: ok=m&sy&((py<row.p_yes) if strict else (py<=row.p_yes))          # someone sells YES into our bid
    else: ok=m&(~sy)&((py>row.p_yes) if strict else (py>=row.p_yes))            # someone buys YES / sells NO into our NO bid
    i=np.argmax(ok) if ok.any() else -1
    return (ts[i]-row.ts) if i>=0 else np.nan
F=te[te.w.isin(SET)].sort_values('ts').drop_duplicates(['w','cid']).copy()      # copy their FIRST fill per market
F['t_fill_strict']=[fill(r,True) for r in F.itertuples()]; F['t_fill_incl']=[fill(r,False) for r in F.itertuples()]
ask=np.where(F.q>0,F.p_yes,1-F.p_yes).clip(0.01,0.99); sh=STAKE/ask; fee=0.05*ask*(1-ask)*sh; win=np.where(F.q>0,F.won,~F.won); F['our_pnl']=np.where(win,sh-STAKE,-STAKE)-fee
F['their_pnl_10']=F.our_pnl            # their P&L on the same fill at $10 (same price, same outcome)
F['their_usd']=F.usd; F['their_pnl_actual']=F.pnl
rows=[]
for w in SET:
    g=F[F.w==w]
    if len(g)==0: continue
    fs=g[g.t_fill_strict.notna()]; fi=g[g.t_fill_incl.notna()]
    rows.append(dict(wallet=name(w),orders=len(g),fill_strict=round(len(fs)/len(g),2),fill_incl=round(len(fi)/len(g),2),med_hours_to_fill=round(fs.t_fill_strict.median()/3600,1) if len(fs) else np.nan,
                     their_roi_first_fills=round(g.their_pnl_10.sum()/(STAKE*len(g)),2),their_pnl_actual=round(g.their_pnl_actual.sum()),our_pnl_strict=round(fs.our_pnl.sum()),our_roi_strict=round(fs.our_pnl.sum()/(STAKE*len(fs)),2) if len(fs) else np.nan,our_pnl_incl=round(fi.our_pnl.sum()),our_roi_incl=round(fi.our_pnl.sum()/(STAKE*len(fi)),2) if len(fi) else np.nan,
                     unfilled_would_have=round(g[g.t_fill_strict.isna()].our_pnl.sum())))
R=pd.DataFrame(rows).sort_values('their_pnl_actual',ascending=False); print("Aug 28 - Sep 18: resting limit at the trader's own price on their first fill per market, $10/order\n"); print(R.to_string(index=False))
tot=F; fs=F[F.t_fill_strict.notna()]; fi=F[F.t_fill_incl.notna()]
print(f"\nALL {len(SET)} wallets: {len(tot)} orders; filled strict {len(fs)/len(tot):.0%}, incl. equal price {len(fi)/len(tot):.0%}; median time to fill {fs.t_fill_strict.median()/3600:.1f} h")
print(f"their ROI on these fills (at $10 each): {tot.their_pnl_10.sum()/(STAKE*len(tot)):+.1%}  |  OUR ROI, filled-strict: {fs.our_pnl.sum()/(STAKE*len(fs)):+.1%} (${fs.our_pnl.sum():+.0f})  |  filled-incl: {fi.our_pnl.sum()/(STAKE*len(fi)):+.1%} (${fi.our_pnl.sum():+.0f})  |  the orders that never filled would have made {tot[tot.t_fill_strict.isna()].our_pnl.sum()/(STAKE*(tot.t_fill_strict.isna().sum())):+.1%}")
print("\nfill rate and our ROI by the trader's entry price (YES-equivalent), strict fills:")
F['band']=pd.cut(F.p_yes,[0,.1,.2,.35,.5,.65,.8,1]); print(F.groupby('band',observed=True).apply(lambda g: pd.Series(dict(orders=len(g),fill=(g.t_fill_strict.notna()).mean(),their_roi=g.their_pnl_10.sum()/(STAKE*len(g)),our_roi=g[g.t_fill_strict.notna()].our_pnl.sum()/(STAKE*max(g.t_fill_strict.notna().sum(),1)),unfilled_roi=g[g.t_fill_strict.isna()].our_pnl.sum()/(STAKE*max(g.t_fill_strict.isna().sum(),1))))).round(2).to_string())
print("\nby side: long YES vs short YES (NO bid):"); F['dir']=np.where(F.q>0,'long','short'); print(F.groupby('dir').apply(lambda g: pd.Series(dict(orders=len(g),fill=(g.t_fill_strict.notna()).mean(),their_roi=g.their_pnl_10.sum()/(STAKE*len(g)),our_roi=g[g.t_fill_strict.notna()].our_pnl.sum()/(STAKE*max(g.t_fill_strict.notna().sum(),1))))).round(2).to_string())
print("\nwinners vs losers: fill rate of orders whose bucket WON vs LOST (adverse selection check):"); F['ours_won']=win; print(F.groupby('ours_won').apply(lambda g: pd.Series(dict(orders=len(g),fill_strict=(g.t_fill_strict.notna()).mean(),fill_incl=(g.t_fill_incl.notna()).mean()))).round(2).to_string())
