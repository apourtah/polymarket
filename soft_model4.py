"""Soft gates v4: v3 + the per-city switches made probabilistic and trained.
Production hard-codes per city (a) whether the agree leg is on, and (b) which model's bucket to buy on disagreement nights
(ridge in Chicago/Houston/Dallas, EWMA in LA/Austin, none in Seattle/Miami) — chosen in-sample. Here every city gets every
leg as a candidate, and the policy has a trained per-city x leg bias c[city,leg] (shrunk to 0 by L2), i.e. a learned degree of
trust in that leg for that city, on top of the Gaussian price/P gates of v3. Legs: agree, ridge (ridge pick on disagreement
nights), ewma (EWMA pick), no_ridge / no_ewma (favorite warmer than that pick, or the bucket 1 warmer than it).
At test time a NO is taken only if its paired YES was taken (a>=0.5), as in production.
Init variants: 'flat' c=0 (no city knowledge) and 'modes' c=+1/-1 from the production MODES; both trained walk-forward.
env: TEST_FROM (2026-03), HURDLE (5), REG (0.05), REG_CITY (0.02), EPOCHS (400), LR (0.05)"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
STAKE=50.0; SLIP=0.01; TEST_FROM=os.environ.get('TEST_FROM','2026-03'); HURDLE=float(os.environ.get('HURDLE','5')); REG=float(os.environ.get('REG','0.05')); REG_CITY=float(os.environ.get('REG_CITY','0.02')); EPOCHS=int(os.environ.get('EPOCHS','400')); LR=float(os.environ.get('LR','0.05'))
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B['month']=B.mday.dt.to_period('M').astype(str)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
CITIES=sorted(MODES); B=B[(B.lo>-999)&(B.hi<999)].copy(); B['pav']=(B.pe+B.pr)/2; B['edge']=B.pav-B.price; B['is_fav']=(B.lo==B.fav).astype(int)
C={}
C['agree']=B[B.agree&(B.lo==B.be)].copy(); C['ridge']=B[~B.agree&(B.lo==B.br)].copy(); C['ewma']=B[~B.agree&(B.lo==B.be)].copy()
for k in ['ridge','ewma']:
    pk=C[k].set_index(['city','mday']); B[f'{k}_lo']=[pk.lo.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B[f'{k}_px']=[pk.price.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]
    n=pd.concat([B[B[f'{k}_lo'].notna()&(B.is_fav==1)&(B.lo>B[f'{k}_lo'])], B[B[f'{k}_lo'].notna()&(B.lo==B[f'{k}_lo']+2)]]).drop_duplicates(['city','mday','lo']).copy()
    n['ypick_px']=n[f'{k}_px']; C['no_'+k]=n
for k,v in C.items(): v['leg']=k; v['side']='no' if k.startswith('no_') else 'yes'
A=pd.concat(C.values()).reset_index(drop=True); A=A[A.price>=0.05].copy()
A['ask']=np.where(A.side=='yes',A.price+SLIP,1-A.price+SLIP).clip(0.01,0.99); sh=STAKE/A.ask; fee=0.05*A.ask*(1-A.ask)*sh; A['winb']=np.where(A.side=='yes',A.won,~A.won); A['pnl']=np.where(A.winb,sh-STAKE,-STAKE)-fee
HARD={'agree':(0.10,0.53),'ridge':(0.05,0.60),'ewma':(0.05,0.60),'no_ridge':(0.35,0.55),'no_ewma':(0.35,0.55)}
FEATS={k:(['price','pav','edge','fav_p']+(['ypick_px'] if k.startswith('no_') else [])) for k in HARD}
def summ(x,size=None):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,px=np.nan,stake=0,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    s=np.ones(len(x)) if size is None else size; p=pd.Series(x.pnl.values*s,index=x.index); m=p.groupby(x.month).sum()
    return pd.Series(dict(n=len(x),win=x.winb.mean(),px=x.ask.mean(),stake=STAKE*s.sum(),pnl=p.sum(),roi=p.sum()/(STAKE*s.sum()),neg_m=(m<0).sum(),worst_m=m.min()))
def mode_on(city,leg): m=MODES[city]; return ('agree' in m) if leg=='agree' else (leg.replace('no_','') in m)
class Gates:
    def __init__(s,leg,tr,init):
        f=FEATS[leg]; s.f=f; lo,hi=HARD[leg]; s.mu=np.array([(lo+hi)/2 if c=='price' else tr[c].mean() for c in f]); s.ls=np.log(np.array([(hi-lo)/2 if c=='price' else max(tr[c].std(),1e-3)*1.5 for c in f]))
        s.w=np.array([1.0 if c=='price' else 0.3 for c in f]); s.b=np.array([1.0]); s.c=np.array([0.0 if init=='flat' else (1.0 if mode_on(ct,leg) else -1.5) for ct in CITIES]); s.init=s.pack().copy(); s.nf=len(f)
    def pack(s): return np.r_[s.mu,s.ls,s.w,s.b,s.c]
    def unpack(s,p): n=s.nf; s.mu,s.ls,s.w,s.b,s.c=p[:n],p[n:2*n],p[2*n:3*n],p[3*n:3*n+1],p[3*n+1:]
    def score(s,X): z=(X[s.f].values-s.mu)/np.exp(s.ls); ci=X.city.map({c:i for i,c in enumerate(CITIES)}).values; return s.b[0]+s.c[ci]+(s.w*(-0.5*z*z)).sum(1),z,ci
    def act(s,X): return 1/(1+np.exp(-s.score(X)[0]))
    def grad(s,X,r):
        sc,z,ci=s.score(X); a=1/(1+np.exp(-sc)); da=a*(1-a)*r; sig=np.exp(s.ls)
        g_b=np.array([da.sum()]); g_w=(da[:,None]*(-0.5*z*z)).sum(0); g_mu=(da[:,None]*(s.w*z/sig)).sum(0); g_ls=(da[:,None]*(s.w*z*z)).sum(0); g_c=np.bincount(ci,weights=da,minlength=len(CITIES))
        p=s.pack(); n=s.nf; reg=np.r_[np.full(3*n+1,REG),np.full(len(CITIES),REG_CITY)]
        obj=(a*r).sum()-(reg*(p-s.init)**2).sum()*len(X); g=np.r_[g_mu,g_ls,g_w,g_b,g_c]-2*reg*(p-s.init)*len(X); return obj,g
def train(leg,tr,init):
    G=Gates(leg,tr,init); r=(tr.pnl-HURDLE).values/STAKE; p=G.pack(); m=np.zeros_like(p); v=np.zeros_like(p); n=G.nf
    for t in range(1,EPOCHS+1):
        G.unpack(p); obj,g=G.grad(tr,r); g=g/len(tr); m=0.9*m+0.1*g; v=0.999*v+0.001*g*g; p=p+LR*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        p[n:2*n]=np.clip(p[n:2*n],np.log(0.01),np.log(2.0)); p[2*n:3*n]=np.clip(p[2*n:3*n],0,10)
    G.unpack(p); return G
months=sorted(A.month.unique()); test_months=[m for m in months if m>=TEST_FROM]; glog=[]
for init in ['flat','modes']:
    A[f'a0_{init}']=np.nan; A[f'a_{init}']=np.nan
    for k in C:
        for mth in test_months:
            tr=A[(A.leg==k)&(A.month<mth)]; te=(A.leg==k)&(A.month==mth)
            if len(tr)<80 or te.sum()==0: continue
            G0=Gates(k,tr,init); A.loc[te,f'a0_{init}']=G0.act(A[te]); G=train(k,tr,init); A.loc[te,f'a_{init}']=G.act(A[te])
            if mth==test_months[-1]: glog.append(dict(init=init,leg=k,n_train=len(tr),mu_price=round(G.mu[0],3),sig_price=round(float(np.exp(G.ls[0])),3),b=round(G.b[0],2),**{f"c_{c[:3]}":round(G.c[i],2) for i,c in enumerate(CITIES)}))
T=A[A.a_flat.notna()&A.a_modes.notna()].copy(); print(f"test months {T.month.min()}..{T.month.max()}, candidates: {T.leg.value_counts().to_dict()}  (hurdle ${HURDLE}, reg {REG}, reg_city {REG_CITY})")
def pair(S,col):
    y=S[S.side=='yes']; out=[y]
    for k in ['ridge','ewma']:
        yn=set(zip(y[y.leg==k].city,y[y.leg==k].mday)); n=S[(S.leg=='no_'+k)&np.array([q in yn for q in zip(S.city,S.mday)])]; out.append(n)
    S=pd.concat(out); S=S.sort_values(col,ascending=False); return pd.concat([S[S.side=='yes'].drop_duplicates(['city','mday']),S[S.side=='no'].drop_duplicates(['city','mday'])])   # one YES, one NO per city-night
def hard(T):
    H=pd.concat([T[(T.leg==k)&T.price.between(*HARD[k])&T.apply(lambda r: mode_on(r.city,k),axis=1)] for k in HARD]); H['col']=1.0; return pair(H,'col')
def soft(T,col,thr=0.5):
    X=T[T[col]>=thr].copy(); X['col']=X[col]; return pair(X,'col')
H=hard(T); res={'hard rule (production modes + bands)':summ(H)}
for init in ['flat','modes']:
    res[f'soft untrained, init={init}']=summ(soft(T,f'a0_{init}')); res[f'soft TRAINED, init={init}']=summ(soft(T,f'a_{init}'))
print("\n### out-of-sample, same months"); print(pd.DataFrame(res).T.round(2).to_string())
S=soft(T,'a_modes'); Sf=soft(T,'a_flat')
print("\nper leg:"); print(pd.DataFrame({**{f"hard {k}":summ(H[H.leg==k]) for k in HARD},**{f"trained(modes) {k}":summ(S[S.leg==k]) for k in HARD},**{f"trained(flat) {k}":summ(Sf[Sf.leg==k]) for k in HARD}}).T.round(2).to_string())
bm=lambda x: x.groupby('month').pnl.sum().round(0); print("\nby month:"); print(pd.DataFrame({'hard':bm(H),'trained(modes)':bm(S),'trained(flat)':bm(Sf)}).T.to_string())
print("by city:"); print(pd.DataFrame({'hard':H.groupby('city').pnl.sum(),'trained(modes)':S.groupby('city').pnl.sum(),'trained(flat)':Sf.groupby('city').pnl.sum()}).round(0).T.to_string())
print("\ntrades per city x leg — hard vs trained(flat) (does the trained city trust rediscover the modes?)")
print(pd.concat([pd.crosstab(H.city,H.leg).add_prefix('hard '),pd.crosstab(Sf.city,Sf.leg).add_prefix('flat ')],axis=1).fillna(0).astype(int).to_string())
print("\nP&L per city x leg, trained(flat):"); print(Sf.pivot_table(index='city',columns='leg',values='pnl',aggfunc='sum').round(0).fillna(0).to_string())
print("\n### trained per-city trust c[city,leg] (last refit; + = more trades in that city for that leg):"); print(pd.DataFrame(glog).set_index(['init','leg']).to_string())
print("\n### threshold sweep on trained(flat) intensity:"); print(pd.DataFrame({f"a>={t:.1f}":summ(soft(T,'a_flat',t)) for t in [0.3,0.5,0.6,0.7,0.8]}).T.round(2).to_string())
