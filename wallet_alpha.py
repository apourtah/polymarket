"""Who has alpha in the highest-temperature markets? Per-wallet, per-day scoring from the taker-side tapes (data/tapes_w).
Every fill is converted to a YES-equivalent position (buy YES / sell NO = long YES; buy NO / sell YES = short YES) and
held to resolution, so each fill has a P&L = q * (won - p_yes). Metrics per wallet: P&L, ROI, markets, days, win rate,
bootstrap t-stat over days (days are the independent unit), markout (YES price move 1h/4h after the fill, from the tape
itself), timing profile, and per-day leaderboard persistence (how often a wallet is top-k on a day). Output: out/wallets_*.csv"""
import pandas as pd, numpy as np, json, os, re, glob, datetime as dt
from zoneinfo import ZoneInfo
from sweep_local import TZ
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_colwidth',44)
CITY_RE=re.compile(r'highest temperature in (.+?) be ')
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)',q)
    return int(mm.group(1)) if mm.group(1) else (-999 if mm.group(3) else int(mm.group(4)))
rows=[]
for f in glob.glob('data/tapes_w/*.json'):
    d=json.load(open(f)); op=d.get('outcome_prices') or []
    if len(op)!=2 or float(op[0]) not in (0.0,1.0): continue
    won=float(op[0])==1.0; city=CITY_RE.search(d['question']).group(1); day=d['end_date'][:10]; lo=bucket(d['question'])
    for t in d['trades']:
        if not t.get('proxyWallet'): continue
        yes_side = (t['outcome']=='Yes')==(t['side']=='BUY')                 # long YES?
        p_yes = t['price'] if t['outcome']=='Yes' else 1-t['price']
        q = t['size'] if yes_side else -t['size']
        rows.append(dict(w=t['proxyWallet'],uname=t.get('name') or t.get('pseudonym') or '',city=city,day=day,lo=lo,cid=f[13:-5],ts=t['timestamp'],side=t['side'],outcome=t['outcome'],price=t['price'],size=t['size'],p_yes=p_yes,q=q,won=won,tx=t['transactionHash']))
T=pd.DataFrame(rows); T['usd']=T['size']*T.price; T['pnl']=T.q*(T.won.astype(float)-T.p_yes)
T['t']=pd.to_datetime(T.ts,unit='s',utc=True); T['local']=[t.tz_convert(TZ[c]) for t,c in zip(T.t,T.city)]; T['lhour']=[x.hour+x.minute/60 for x in T.local]
T['mday']=pd.to_datetime(T.day); T['rel_day']=[(x.date()-m.date()).days for x,m in zip(T.local,T.mday)]   # -1 = day before, 0 = market day
T.to_parquet('out/wallet_fills.parquet')
print(f"{len(T):,} taker fills, {T.w.nunique():,} wallets, {T.cid.nunique():,} markets, {T.day.nunique()} days ({T.day.min()}..{T.day.max()})")
# ---------- markout: YES price 1h / 4h after each fill, from the same market's later prints ----------
T=T.sort_values(['cid','ts']); mo={}
for cid,g in T.groupby('cid'):
    ts=g.ts.values; py=g.p_yes.values
    for h,col in [(3600,'mo1h'),(4*3600,'mo4h')]:
        idx=np.searchsorted(ts,ts+h,side='right')-1; later=np.where(idx>np.arange(len(ts)),py[np.minimum(idx,len(ts)-1)],np.nan)
        mo.setdefault(col,[]).append(pd.Series(later,index=g.index))
for col in mo: T[col]=pd.concat(mo[col]).reindex(T.index)
T['mk1h']=np.sign(T.q)*(T.mo1h-T.p_yes); T['mk4h']=np.sign(T.q)*(T.mo4h-T.p_yes)      # in the direction of the trade, per share
# ---------- per wallet ----------
def wstats(g):
    days=g.groupby('day').pnl.sum(); usd=g.usd.sum()
    boot=[0.0]; rng=np.random.default_rng(0); dv=days.values
    if len(days)>=8:
        boot=rng.choice(dv,(300,len(dv))).sum(1)
    tstat=days.mean()/(days.std(ddof=1)/np.sqrt(len(days))) if len(days)>2 and days.std()>0 else np.nan
    mk=g.dropna(subset=['mk1h']); 
    return pd.Series(dict(fills=len(g),markets=g.cid.nunique(),days=len(days),usd=usd,pnl=g.pnl.sum(),roi=g.pnl.sum()/usd if usd else np.nan,
        day_win=(days>0).mean(),tstat=tstat,p_boot=(np.array(boot)<=0).mean(),mk1h=np.average(mk.mk1h,weights=mk.usd) if len(mk) else np.nan,mk4h=np.average(g.dropna(subset=['mk4h']).mk4h,weights=g.dropna(subset=['mk4h']).usd) if g.mk4h.notna().any() else np.nan,
        share_long=(g.q>0).mean(),avg_p=np.average(g.p_yes,weights=g.usd),med_size=g['size'].median(),share_daybefore=(g.rel_day<0).mean(),cities=g.city.nunique(),uname=g.uname.iloc[0],first=g.day.min(),last=g.day.max()))
W=T.groupby('w').apply(wstats); W.to_csv('out/wallets_all.csv')
print(f"\nwallets with >= 10 markets and >= 8 days: {((W.markets>=10)&(W.days>=8)).sum()}")
A=W[(W.markets>=10)&(W.days>=8)].copy()
print("\n### top 25 by P&L (>=10 markets, >=8 days)"); print(A.sort_values('pnl',ascending=False).head(25)[['fills','markets','days','usd','pnl','roi','day_win','tstat','p_boot','mk1h','mk4h','share_long','avg_p','med_size','share_daybefore','uname']].round(3).to_string())
print("\n### top 25 by bootstrap significance (p_boot small = consistent across days), pnl > 500"); print(A[A.pnl>500].sort_values(['p_boot','tstat'],ascending=[True,False]).head(25)[['fills','markets','days','usd','pnl','roi','day_win','tstat','p_boot','mk1h','mk4h','avg_p','med_size','share_daybefore','uname']].round(3).to_string())
print("\n### top 20 by 1h markout (informed flow: price keeps moving their way), usd >= 5000"); print(A[A.usd>=5000].sort_values('mk1h',ascending=False).head(20)[['fills','markets','days','usd','pnl','roi','mk1h','mk4h','avg_p','med_size','share_daybefore','uname']].round(3).to_string())
print("\n### bottom 10 by P&L (the liquidity)"); print(A.sort_values('pnl').head(10)[['fills','markets','days','usd','pnl','roi','day_win','mk1h','avg_p','uname']].round(3).to_string())
# ---------- per-day leaderboard persistence ----------
D=T.groupby(['day','w']).agg(pnl=('pnl','sum'),usd=('usd','sum')).reset_index(); D['rank']=D.groupby('day').pnl.rank(ascending=False,method='first'); ndays=D.day.nunique()
top=D[D['rank']<=10].groupby('w').agg(top10_days=('day','size'),pnl=('pnl','sum')); top['days_active']=D.groupby('w').day.nunique(); top['top10_rate']=top.top10_days/top.days_active
print(f"\n### per-day top-10 persistence ({ndays} days): wallets most often in a day's top-10 by P&L"); print(top[top.days_active>=8].sort_values('top10_days',ascending=False).head(20).round(2).to_string())
# split-half persistence
days=sorted(T.day.unique()); odd=set(days[::2]); P1=T[T.day.isin(odd)].groupby('w').pnl.sum(); P2=T[~T.day.isin(odd)].groupby('w').pnl.sum(); both=W[(W.markets>=10)].index
c=pd.concat([P1.reindex(both),P2.reindex(both)],axis=1).dropna(); c.columns=['a','b']; print(f"\nsplit-half persistence (odd vs even days, wallets >=10 markets, n={len(c)}): Spearman {c.corr(method='spearman').iloc[0,1]:.3f}; of the top-20 on odd days, {(c.loc[c.a.nlargest(20).index,'b']>0).sum()} are positive on even days, of the bottom-20 {(c.loc[c.a.nsmallest(20).index,'b']<0).sum()} stay negative")
# ---------- timing of the profitable wallets vs everyone ----------
A20=A.sort_values('pnl',ascending=False).head(20).index; T['grp']=np.where(T.w.isin(A20),'top20','rest'); T=T.drop(columns=['uname'])
T['band']=pd.cut(T.lhour,[-1,3,6,9,12,15,18,21,24],labels=['00-03','03-06','06-09','09-12','12-15','15-18','18-21','21-24'])
print("\n### when do they trade? share of $ by local hour band x day (-1 = day before), top-20 wallets vs rest"); print((pd.pivot_table(T,index=['grp','rel_day'],columns='band',values='usd',aggfunc='sum',observed=True).div(T.groupby('grp').usd.sum(),axis=0,level=0)*100).round(1).loc[(slice(None),[-1,0]),:].to_string())
T.to_parquet('out/wallet_fills.parquet')
