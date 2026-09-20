"""Which wallets are the same person? Signals: (1) default-name creation timestamps minutes apart, (2) fills in the same market
within seconds of each other, repeatedly (co-activity far above chance), (3) near-identical behavioural fingerprint (local-hour
histogram, city mix, side, price level, clip-size signature) with (4) non-overlapping active periods (hand-off) or overlapping.
Input out/wallet_fills.parquet, out/wallets_all.csv. Output out/wallet_pairs.csv, out/wallet_clusters.csv"""
import pandas as pd, numpy as np, re, itertools
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_colwidth',30)
T=pd.read_parquet('out/wallet_fills.parquet'); W=pd.read_csv('out/wallets_all.csv',index_col=0)
U=W[(W.fills>=20)].index; T=T[T.w.isin(U)].copy(); print(f"{len(U)} wallets with >= 20 fills")
# ---- fingerprint ----
T['hb']=(T.lhour//1).astype(int); hours=pd.crosstab(T.w,T.hb).reindex(columns=range(24),fill_value=0); hours=hours.div(hours.sum(1),axis=0)
city=pd.crosstab(T.w,T.city); city=city.div(city.sum(1),axis=0)
T['sz_int']=(T['size']%1==0); T['sz_r5']=(T['size']%5==0); T['sz_dec']=T['size'].round(2).astype(str).str.split('.').str[1].str.len()
sz=T.groupby('w').agg(med_size=('size','median'),int_share=('sz_int','mean'),r5_share=('sz_r5','mean'),share_long=('q',lambda s:(s>0).mean()),avg_p=('p_yes','mean'),rel_before=('rel_day',lambda s:(s<0).mean()),first=('day','min'),last=('day','max'),n=('size','size'))
topsz=T.groupby('w')['size'].agg(lambda s: set(s.round(2).value_counts().head(5).index))
def cos(A): A=A.values; n=np.linalg.norm(A,axis=1,keepdims=True)+1e-9; return (A/n)@(A/n).T
Hc=cos(hours); Cc=cos(city); idx=list(hours.index); pos={w:i for i,w in enumerate(idx)}
# ---- creation timestamps from default names "0x...-<epoch ms>" ----
cr={}
for w,nm in W.uname.items():
    if isinstance(nm,str) and re.match(r'^0x[0-9a-fA-F]{40}-\d{13}$',nm): cr[w]=int(nm.split('-')[1])/1000
print(f"{len(cr)} wallets carry a creation timestamp in their default name")
# ---- co-activity: same market, fills within 30 s, counted per pair ----
T=T.sort_values(['cid','ts']); co={}; ex={}
for cid,g in T.groupby('cid'):
    ws=g.w.values; ts=g.ts.values; span=max(ts[-1]-ts[0],3600); cnt=g.w.value_counts()
    for i in range(len(g)):
        j=i+1
        while j<len(g) and ts[j]-ts[i]<=30:
            if ws[i]!=ws[j]: k=tuple(sorted((ws[i],ws[j]))); co[k]=co.get(k,0)+1
            j+=1
    # expected co-fills if the two wallets' fills were independent uniform over the market's active span
    big=cnt[cnt>=3].index
    for a in big:
        for b in big:
            if a<b: ex[(a,b)]=ex.get((a,b),0)+cnt[a]*cnt[b]*60/span
co=pd.Series(co).sort_values(ascending=False); print(f"{len(co)} wallet pairs ever co-fill within 30 s")
f=T.groupby('w').size(); pairs=[]
for (a,b),n in co.items():
    if n<5: continue
    pairs.append(dict(a=a,b=b,cofills=n,ratio=n/max(ex.get((a,b),0.05),0.05)))
P=pd.DataFrame(pairs); ratio=P.set_index(['a','b']).ratio.to_dict()
# ---- fingerprint similarity for candidate pairs (co-active OR created within 1 h OR both in top-300 by fills) ----
top300=set(f.sort_values(ascending=False).head(300).index)
cand=set(zip(P.a,P.b)) if len(P) else set()
crs=sorted(cr.items(),key=lambda x:x[1])
for (w1,t1),(w2,t2) in zip(crs,crs[1:]):
    if t2-t1<=3600 and w1 in pos and w2 in pos: cand.add(tuple(sorted((w1,w2))))
for a,b in itertools.combinations(sorted(top300),2): cand.add((a,b))
rows=[]
for a,b in cand:
    if a not in pos or b not in pos: continue
    i,j=pos[a],pos[b]; sa,sb=sz.loc[a],sz.loc[b]
    overlap=not (sa['last']<sb['first'] or sb['last']<sa['first'])
    rows.append(dict(a=a,b=b,hour_cos=Hc[i,j],city_cos=Cc[i,j],size_jacc=len(topsz[a]&topsz[b])/max(len(topsz[a]|topsz[b]),1),d_long=abs(sa.share_long-sb.share_long),d_p=abs(sa.avg_p-sb.avg_p),d_before=abs(sa.rel_before-sb.rel_before),
                     overlap=overlap,created_gap_h=(abs(cr[a]-cr[b])/3600 if a in cr and b in cr else np.nan),cofills=int(co.get((a,b),0)),co_ratio=ratio.get((a,b),np.nan),
                     name_a=W.uname.get(a,''),name_b=W.uname.get(b,''),pnl_a=W.pnl[a],pnl_b=W.pnl[b],fills_a=int(f[a]),fills_b=int(f[b])))
R=pd.DataFrame(rows); R['fp']=(R.hour_cos+R.city_cos+R.size_jacc+(1-R.d_long)+(1-R.d_p*2).clip(0,1)+(1-R.d_before))/6
R.to_csv('out/wallet_pairs.csv',index=False)
print("\n### A. wallets created within 1 hour of each other (default-name timestamps) — strongest same-person signal")
a=R[R.created_gap_h<=1].sort_values('created_gap_h'); print(a[['a','b','created_gap_h','fp','hour_cos','city_cos','size_jacc','overlap','cofills','pnl_a','pnl_b']].round(3).head(25).to_string(index=False))
print("\n### B. named siblings (same name stem)")
nm=W.uname.dropna(); stem=nm[~nm.str.match(r'^0x')].str.lower().str.replace(r'[^a-z]','',regex=True).str.replace(r'(high|low|temp|tation|bot|weather|wx)','',regex=True)
sib=nm.groupby(stem).apply(list); print([ (k,v) for k,v in sib.items() if len(v)>1 and k][:15])
print("\n### C. co-activity (same market, fills within 30 s) vs chance (co_ratio = observed / expected under independence), cofills >= 30")
c=R[(R.cofills>=30)].sort_values('co_ratio',ascending=False); print(c[['a','b','cofills','co_ratio','fp','hour_cos','size_jacc','overlap','name_a','name_b','pnl_a','pnl_b']].round(2).head(25).to_string(index=False))
print("\n### D. near-identical fingerprints (fp >= 0.9) among the top-300 by activity — overlapping (parallel wallets) and non-overlapping (hand-offs)")
d=R[(R.fp>=0.9)&(R.a.isin(top300))&(R.b.isin(top300))].sort_values('fp',ascending=False); print(d[['a','b','fp','hour_cos','city_cos','size_jacc','overlap','cofills','name_a','name_b','pnl_a','pnl_b','fills_a','fills_b']].round(3).head(30).to_string(index=False))
# clusters = connected components over strong links
strong=R[(R.created_gap_h<=1)|((R.cofills>=30)&(R.co_ratio>=8)&(R.fp>=0.85))|((R.fp>=0.95)&(R.hour_cos>=0.97)&(R.size_jacc>=0.6))]
par={}
def find(x):
    while par.setdefault(x,x)!=x: par[x]=par[par[x]]; x=par[x]
    return x
for a,b in zip(strong.a,strong.b): par[find(a)]=find(b)
grp={}
for x in list(par): grp.setdefault(find(x),[]).append(x)
comps=[sorted(c) for c in grp.values() if len(c)>=2]
rows=[]
for k,c in enumerate(sorted(comps,key=lambda c:-W.pnl.reindex(c).sum())):
    rows.append(dict(cluster=k,n=len(c),pnl=W.pnl.reindex(c).sum(),usd=W.usd.reindex(c).sum(),wallets=','.join(x[:10] for x in c),names=','.join(str(W.uname.get(x,''))[:14] for x in c)))
CL=pd.DataFrame(rows); CL.to_csv('out/wallet_clusters.csv',index=False); print(f"\n### E. clusters (connected components of strong links): {len(CL)}"); print(CL.head(20)[['cluster','n','pnl','usd','names']].round(0).to_string(index=False))
ordered=sorted(comps,key=lambda c:-W.pnl.reindex(c).sum()); memb={w:k for k,c in enumerate(ordered) for w in c}
A=W[(W.markets>=10)&(W.days>=8)].sort_values('pnl',ascending=False).head(25); print("\n### F. top-25 alpha wallets: are they in a cluster?"); print(pd.DataFrame({'pnl':A.pnl.round(0),'name':A.uname,'cluster':[memb.get(w,'-') for w in A.index],'cluster_size':[len(ordered[memb[w]]) if w in memb else 1 for w in A.index]}).to_string())
