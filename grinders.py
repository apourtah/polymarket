"""Reverse-engineer the high-volume, small-margin wallets ('grinders'): what, when, at what price, in what structure
(single bucket / multi-bucket portfolios / round trips), relative to the favorite and the winner, and how they make money."""
import pandas as pd, numpy as np, json, re
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
T=pd.read_parquet('out/wallet_fills.parquet').sort_values('ts'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
G={'0xd25156e222c9b907b128e27c36821fdb41db4d37':'neo7777','0xbbb72a812cfbc5217d77c0a0018c71f174d3a11a':'sailor82','0x122cb94c437ea5e6f088c6c0c143c592ab8efbed':'0x122cb9','0x0484bf76425db5021ce2d369f78fc125906d4830':'sleeper-service','0xec87cb382fd131f2af9abf11840752df74fd43f9':'Rexc8','0xe6307b98bceee038834a68358e17804999c6ef20':'anonymous5474495','0x6011655c4afb76f36dd1b08a137a1ba73466b31e':'HighTempTation'}
# favorite at fill time: last print per bucket in the same (city, day) event before ts
T['ev']=T.city+'|'+T.day; last={}; fav_lo=[]; fav_p=[]; nb=[]
for ev,g in T.groupby('ev',sort=False):
    pass
T=T.sort_values('ts'); state={}; FL=np.empty(len(T)); FP=np.empty(len(T)); i=0
for r in T.itertuples():
    s=state.setdefault(r.ev,{})
    if s: b=max(s,key=s.get); FL[i]=b; FP[i]=s[b]
    else: FL[i]=np.nan; FP[i]=np.nan
    s[r.lo]=r.p_yes; i+=1
T['fav_lo']=FL; T['fav_p']=FP
win={ (c,d):lo for (c,d,lo),w in T.groupby(['city','day','lo']).won.first().items() if w}
T['win_lo']=[win.get((c,d)) for c,d in zip(T.city,T.day)]; T['off_fav']=(T.lo-T.fav_lo)/2; T['off_win']=(T.lo-T.win_lo)/2
T['minute']=(T.ts%3600)//60; T['sec_in_min']=T.ts%60
for w,name in G.items():
    g=T[T.w==w].copy()
    if len(g)==0: continue
    print(f"\n{'='*110}\n{name}  {w}  fills {len(g)}  markets {g.cid.nunique()}  events {g.ev.nunique()}  $ {g.usd.sum():,.0f}  P&L(hold) {g.pnl.sum():+,.0f}  ROI {g.pnl.sum()/g.usd.sum():+.1%}")
    print("side/outcome mix:", g.groupby(['side','outcome']).size().to_dict(), "| long-YES-equivalent share of fills %.0f%%, of $ %.0f%%"%(100*(g.q>0).mean(),100*g.usd[g.q>0].sum()/g.usd.sum()))
    print("timing: day-before %.0f%% of $; local-hour $ share:"%(100*g.usd[g.rel_day<0].sum()/g.usd.sum()), (g.groupby(pd.cut(g.lhour,[-1,3,6,9,12,15,18,21,24],labels=['0-3','3-6','6-9','9-12','12-15','15-18','18-21','21-24'])).usd.sum()/g.usd.sum()*100).round(0).to_dict())
    print("minute-of-hour concentration (share in :53-:01): %.0f%%   seconds-in-minute std %.0f"%(100*((g.minute>=53)|(g.minute<=1)).mean(), g.sec_in_min.std()))
    print("price of what they buy (YES-equivalent p): quantiles", g.p_yes.quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
    lb=g[g.q>0]; sb=g[g.q<0]; print("  long fills: p_yes med %.2f, mean %.2f | short fills (buy NO / sell YES): p_yes med %.2f -> NO price med %.2f"%(lb.p_yes.median() if len(lb) else np.nan,lb.p_yes.mean() if len(lb) else np.nan,sb.p_yes.median() if len(sb) else np.nan,1-sb.p_yes.median() if len(sb) else np.nan))
    print("bucket vs favorite at the time (+ = warmer):", g.off_fav.clip(-3,3).value_counts(normalize=True).sort_index().round(2).to_dict())
    print("bucket vs eventual winner:", g.off_win.clip(-3,3).value_counts(normalize=True).sort_index().round(2).to_dict())
    print("P&L by (long/short) x bucket-vs-favorite:"); g['dir']=np.where(g.q>0,'long','short'); print(pd.pivot_table(g,index='dir',columns=g.off_fav.clip(-2,2),values='pnl',aggfunc='sum').round(0).fillna(0).to_string())
    print("P&L by YES-equivalent price band x dir:"); print(pd.pivot_table(g,index='dir',columns=pd.cut(g.p_yes,[0,.05,.15,.3,.5,.7,.85,.95,1]),values='pnl',aggfunc=['sum','size'],observed=True).round(0).to_string())
    # structure per event: how many buckets, both sides?, cost vs guaranteed payout (multi-bucket arbitrage?)
    E=g.groupby('ev').agg(n_buckets=('lo','nunique'),n_fills=('lo','size'),usd=('usd','sum'),pnl=('pnl','sum'),long_b=('q',lambda s:(s>0).sum()),short_b=('q',lambda s:(s<0).sum()))
    print("per event: buckets touched", E.n_buckets.value_counts().sort_index().head(8).to_dict(), "| fills/event med %.0f | events with both long and short %.0f%%"%(E.n_fills.median(),100*((E.long_b>0)&(E.short_b>0)).mean()))
    # guaranteed-payout check: net YES-equivalent position per bucket in the event; payout under each outcome
    arb=0; tot=0; worst=[]
    for ev,ge in g.groupby('ev'):
        pos=ge.groupby('lo').q.sum(); cost=(ge.q*ge.p_yes).sum()           # cost of long-YES-equivalent (shorts contribute negative cost = they received credit) 
        outcomes=set(pos.index)|{ge.win_lo.iloc[0]}; payouts={o:pos.get(o,0.0) for o in outcomes}; payouts['other']=0.0
        mn=min(payouts.values()); worst.append(mn-cost); tot+=1; arb+= (mn-cost)>0
    print("events where the portfolio's WORST-case payoff still beats its cost (riskless structure): %d of %d (%.0f%%); median worst-case P&L per event $%.0f"%(arb,tot,100*arb/tot,np.median(worst)))
    # round trips: buy then sell the same token within the event
    rt=0; rtp=0.0
    for (ev,lo),gg in g.groupby(['ev','lo']):
        if (gg.q>0).any() and (gg.q<0).any():
            rt+=1; buys=gg[gg.q>0]; sells=gg[gg.q<0]; rtp+=min(buys.q.sum(),-sells.q.sum())*(sells.p_yes.mean()-buys.p_yes.mean())
    print("markets with both buys and sells (round trips): %d of %d; realised spread P&L from round trips ~$%.0f"%(rt,g.groupby(['ev','lo']).ngroups,rtp))
    # what happens to the price after their fill: markout by horizon
    print("markout in their direction: 1h %+.3f  4h %+.3f (per share), win rate of fills %.0f%%"%(g.mk1h.mean(),g.mk4h.mean(),100*(((g.q>0)&g.won)|((g.q<0)&~g.won)).mean()))
    # size & cadence
    gaps=g.groupby('day').ts.diff().dropna(); print("size: med %.1f sh, $ med %.0f; fills/day med %.0f; inter-fill gap med %.0f s, share <5 s %.0f%%"%(g['size'].median(),g.usd.median(),g.groupby('day').size().median(),gaps.median(),100*(gaps<5).mean()))
    # daily P&L consistency
    d=g.groupby('day').pnl.sum(); print("daily P&L: mean %+.0f sd %.0f, positive days %.0f%%, worst %+.0f"%(d.mean(),d.std(),100*(d>0).mean(),d.min()))
T.to_parquet('out/wallet_fills_ctx.parquet')
