"""The real winners are makers. For the top realised-P&L wallets: maker share, both-sides structure, hours, city breadth,
spread capture (VWAP sold - VWAP bought per market), directional skew (net inventory at resolution), verified against the
official user-pnl curve. Uses out/real_winners.csv, data/realized/, the taker tape and data-api /activity (last 5000 rows)."""
import pandas as pd, numpy as np, json, requests, time, re, os
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_colwidth',30)
S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
Wr=pd.read_csv('out/real_winners.csv',index_col=0); T=pd.read_parquet('out/wallet_fills.parquet')
tb=T[T.side=='BUY'].groupby('w')['size'].sum(); Wr['taker_buy_sh']=tb.reindex(Wr.index).fillna(0); Wr['maker_share']=(1-Wr.taker_buy_sh/Wr.bought_sh).clip(0,1)
A=Wr[(Wr.markets>=10)&(Wr.days>=8)]; TOP=A.sort_values('rpnl',ascending=False).head(20)
print("### top-20 by realised P&L, maker share = 1 - taker-bought shares / total bought shares"); print(TOP[['markets','days','cost','rpnl','roi','day_win','t','tape_pnl','maker_share','name']].round(2).to_string())
print("\nrealised profit of all positive wallets (>=10 mkts): $%.0f; share earned by wallets with maker_share >= 0.5: %.0f%%; >= 0.8: %.0f%%"%(A[A.rpnl>0].rpnl.sum(),100*A[(A.rpnl>0)&(A.maker_share>=0.5)].rpnl.sum()/A[A.rpnl>0].rpnl.sum(),100*A[(A.rpnl>0)&(A.maker_share>=0.8)].rpnl.sum()/A[A.rpnl>0].rpnl.sum()))
print("realised LOSS of all negative wallets: $%.0f; share lost by wallets with maker_share <= 0.2 (takers): %.0f%%"%(A[A.rpnl<0].rpnl.sum(),100*A[(A.rpnl<0)&(A.maker_share<=0.2)].rpnl.sum()/A[A.rpnl<0].rpnl.sum()))
print("\nby maker-share bucket (wallets >= 10 markets): n, total realised P&L, median ROI, share of wallets positive")
print(A.groupby(pd.cut(A.maker_share,[-0.01,0.2,0.5,0.8,1.0]),observed=True).agg(n=('rpnl','size'),rpnl=('rpnl','sum'),med_roi=('roi','median'),pos=('rpnl',lambda s:(s>0).mean()),cost=('cost','sum')).round(2).to_string())
# official check + activity structure for the top 8
rows=[]
for w in TOP.index[:8]:
    try:
        j=S.get("https://user-pnl-api.polymarket.com/user-pnl",params={"user_address":w,"interval":"all","fidelity":"1d"},timeout=60).json(); p=pd.DataFrame(j); p['t']=pd.to_datetime(p.t,unit='s'); off=p[p.t>='2026-08-07'].p.iloc[0]; end=p[p.t<='2026-09-19'].p.iloc[-1]; official=end-off; start=str(p.t.min().date())
    except Exception: official=np.nan; start=''
    acts=[]; o=0
    while o<=4500:
        r=S.get("https://data-api.polymarket.com/activity",params={"user":w,"limit":500,"offset":o,"type":"TRADE"},timeout=60).json()
        if not isinstance(r,list) or not r: break
        acts+=r; o+=500
        if r[-1]['timestamp']<1786000000: break
        time.sleep(0.15)
    Ac=pd.DataFrame(acts)
    if len(Ac):
        Ac['t']=pd.to_datetime(Ac.timestamp,unit='s',utc=True); Ac=Ac[Ac.t>='2026-08-07']; wx=Ac[Ac.title.str.contains('temperature',case=False,na=False)]
        cities=wx.title.str.extract(r'temperature in (.+?) be')[0]; both=wx.groupby('conditionId').outcome.nunique(); sides=wx.groupby('conditionId').side.nunique()
        vw=[]
        for cid,g in wx.groupby('conditionId'):
            b=g[g.side=='BUY']; s_=g[g.side=='SELL']
            if len(b) and len(s_): vw.append((s_.usdcSize.sum()/s_['size'].sum())-(b.usdcSize.sum()/b['size'].sum()))
        rows.append(dict(w=w[:10],name=TOP.loc[w,'name'],official_6w=round(official) if official==official else None,since=start,act_rows=len(Ac),weather_share=round(len(wx)/len(Ac),2),n_cities=cities.nunique(),buy_sell=f"{int((wx.side=='BUY').sum())}/{int((wx.side=='SELL').sum())}",both_outcomes=round((both>1).mean(),2),both_sides=round((sides>1).mean(),2),vwap_spread=round(np.median(vw),3) if vw else None,med_size=round(wx['size'].median(),1),hours_active=int(wx.t.dt.hour.nunique()),window=f"{wx.t.min().date()}..{wx.t.max().date()}"))
    time.sleep(0.3)
print("\n### top-8: official 6-week P&L (user-pnl API), activity structure (last 5000 trades; both_outcomes = share of markets where they traded YES and NO; vwap_spread = median sell VWAP - buy VWAP per market)")
print(pd.DataFrame(rows).to_string(index=False))
# per-position view for the #1: do they hold both outcomes at resolution? distribution of position ROI
w=TOP.index[0]; P=pd.DataFrame(json.load(open(f'data/realized/{w}.json'))); P=P[P.title.str.contains('highest temperature',case=False,na=False)&(P.endDate>='2026-08-07')]; P['cost']=P.totalBought*P.avgPrice; P['roi']=P.realizedPnl/P.cost
print(f"\n### #1 {TOP.loc[w,'name']}: {len(P)} closed positions; outcomes held per market:", P.groupby('conditionId').outcome.nunique().value_counts().to_dict())
print("position ROI distribution:", P.roi.describe()[['25%','50%','75%']].round(2).to_dict(), "| share of positions profitable %.0f%% | avg buy price YES %.2f NO %.2f"%(100*(P.realizedPnl>0).mean(),P[P.outcome=='Yes'].avgPrice.mean(),P[P.outcome=='No'].avgPrice.mean()))
print("P&L by outcome held:", P.groupby('outcome').agg(n=('realizedPnl','size'),cost=('cost','sum'),pnl=('realizedPnl','sum')).round(0).to_dict())
