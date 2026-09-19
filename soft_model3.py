"""Soft gates v3: the DECISION POLICY is trained end-to-end on realised P&L (not likelihood + fixed EV trigger).

Policy per leg L and candidate x:  s = b_L + sum_i w_{L,i} * ( -(x_i - mu_{L,i})^2 / (2 sigma_{L,i}^2) )      (log-product of Gaussian gates)
                                   a = sigmoid(s)  in [0,1]  = trade intensity (train: soft; test: trade if a >= 0.5, or size = a)
Objective (train window):  maximise  sum_t a_t * (pnl_t - HURDLE)  - REG * ||params - init||^2
  HURDLE = $ per trade the trade must clear (a capital charge; 0.10*STAKE = 10% ROI hurdle), so the policy learns to take only
  trades whose conditional expectation beats it — the trigger threshold is therefore trained, not set.
Gradients are analytic; optimiser = Adam; init = the production hard bands (centre = band midpoint, sigma = half-width) so the
untrained policy ~ the current rule, and training moves it only where P&L says so.
Walk-forward: refit at each month start on all earlier candidate nights, test on that month (no look-ahead).
Candidates = production leg structure without price bands (agree pick in agree cities; ridge pick in ridge cities; EWMA pick in
ewma cities; NO_A favorite warmer than the disagree YES pick; NO_B bucket 1 warmer than the ridge pick), price >= 0.05.
Features per candidate: price, pav=(Pe+Pr)/2, edge=pav-price, fav_p; NO legs also ypick_px (the paired YES price).
env: TEST_FROM (2026-03), HURDLE (5), REG (0.02), EPOCHS (400), LR (0.05)"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore'); pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
STAKE=50.0; SLIP=0.01; TEST_FROM=os.environ.get('TEST_FROM','2026-03'); HURDLE=float(os.environ.get('HURDLE','5')); REG=float(os.environ.get('REG','0.02')); EPOCHS=int(os.environ.get('EPOCHS','400')); LR=float(os.environ.get('LR','0.05'))
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B['month']=B.mday.dt.to_period('M').astype(str)
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B=B[(B.lo>-999)&(B.hi<999)].copy(); B['pav']=(B.pe+B.pr)/2; B['edge']=B.pav-B.price; B['is_fav']=(B.lo==B.fav).astype(int)
B['mode_agree']=B.city.map(lambda c: int('agree' in MODES[c])); B['mode_ridge']=B.city.map(lambda c: int('ridge' in MODES[c])); B['mode_ewma']=B.city.map(lambda c: int('ewma' in MODES[c]))
C={}
C['agree']=B[B.agree&(B.lo==B.be)&(B.mode_agree==1)].copy(); C['dis_ridge']=B[~B.agree&(B.lo==B.br)&(B.mode_ridge==1)].copy(); C['dis_ewma']=B[~B.agree&(B.lo==B.be)&(B.mode_ewma==1)].copy()
yp=pd.concat([C['dis_ridge'].assign(leg='dis_ridge'),C['dis_ewma'].assign(leg='dis_ewma')]).drop_duplicates(['city','mday']).set_index(['city','mday'])
B['ypick_lo']=[yp.lo.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['ypick_px']=[yp.price.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['ypick_leg']=[yp.leg.get((c,d)) for c,d in zip(B.city,B.mday)]
C['no']=pd.concat([B[B.ypick_lo.notna()&(B.is_fav==1)&(B.lo>B.ypick_lo)],B[(B.ypick_leg=='dis_ridge')&(B.lo==B.br+2)]]).drop_duplicates(['city','mday','lo']).copy()
for k,v in C.items(): v['leg']=k; v['side']='no' if k=='no' else 'yes'
A=pd.concat(C.values()); A=A[A.price>=0.05].copy()
A['ask']=np.where(A.side=='yes',A.price+SLIP,1-A.price+SLIP).clip(0.01,0.99); sh=STAKE/A.ask; fee=0.05*A.ask*(1-A.ask)*sh; A['winb']=np.where(A.side=='yes',A.won,~A.won)
A['pnl']=np.where(A.winb,sh-STAKE,-STAKE)-fee
HARD={'agree':(0.10,0.50),'dis_ridge':(0.05,0.60),'dis_ewma':(0.05,0.60),'no':(0.35,0.55)}
FEATS={'agree':['price','pav','edge','fav_p'],'dis_ridge':['price','pav','edge','fav_p'],'dis_ewma':['price','pav','edge','fav_p'],'no':['price','pav','edge','fav_p','ypick_px']}
def summ(x,size=None):
    if len(x)==0: return pd.Series(dict(n=0,win=np.nan,px=np.nan,stake=0,pnl=0,roi=np.nan,neg_m=np.nan,worst_m=np.nan))
    s=np.ones(len(x)) if size is None else size; p=pd.Series(x.pnl.values*s,index=x.index); m=p.groupby(x.month).sum()
    return pd.Series(dict(n=len(x),win=x.winb.mean(),px=x.ask.mean(),stake=STAKE*s.sum(),pnl=p.sum(),roi=p.sum()/(STAKE*s.sum()),neg_m=(m<0).sum(),worst_m=m.min()))
# ---------- policy ----------
class Gates:
    def __init__(s,leg,tr):
        f=FEATS[leg]; s.f=f; lo,hi=HARD[leg]; s.mu=np.array([ (lo+hi)/2 if c=='price' else tr[c].mean() for c in f]); s.ls=np.log(np.array([ (hi-lo)/2 if c=='price' else max(tr[c].std(),1e-3)*1.5 for c in f]))
        s.w=np.array([1.0 if c=='price' else 0.3 for c in f]); s.b=np.array([1.0]); s.init=s.pack().copy()
    def pack(s): return np.r_[s.mu,s.ls,s.w,s.b]
    def unpack(s,p): n=len(s.f); s.mu,s.ls,s.w,s.b=p[:n],p[n:2*n],p[2*n:3*n],p[3*n:]
    def score(s,X): z=(X[s.f].values-s.mu)/np.exp(s.ls); return s.b[0]+(s.w*(-0.5*z*z)).sum(1), z
    def act(s,X): return 1/(1+np.exp(-s.score(X)[0]))
    def grad(s,X,r):                       # maximise sum a*(r) - REG*||p-init||^2 ; returns (objective, grad)
        sc,z=s.score(X); a=1/(1+np.exp(-sc)); da=a*(1-a)*r; sig=np.exp(s.ls)
        g_b=np.array([da.sum()]); g_w=(da[:,None]*(-0.5*z*z)).sum(0); g_mu=(da[:,None]*(s.w*z/sig)).sum(0); g_ls=(da[:,None]*(s.w*z*z)).sum(0)
        p=s.pack(); obj=(a*r).sum()-REG*((p-s.init)**2).sum()*len(X); g=np.r_[g_mu,g_ls,g_w,g_b]-2*REG*(p-s.init)*len(X); return obj,g
def train(leg,tr):
    G=Gates(leg,tr); r=(tr.pnl-HURDLE).values/STAKE; p=G.pack(); m=np.zeros_like(p); v=np.zeros_like(p)
    for t in range(1,EPOCHS+1):
        G.unpack(p); obj,g=G.grad(tr,r); g=g/len(tr)          # ascent
        m=0.9*m+0.1*g; v=0.999*v+0.001*g*g; p=p+LR*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
        p[len(G.f):2*len(G.f)]=np.clip(p[len(G.f):2*len(G.f)],np.log(0.01),np.log(2.0)); p[2*len(G.f):3*len(G.f)]=np.clip(p[2*len(G.f):3*len(G.f)],0,10)
    G.unpack(p); return G
months=sorted(A.month.unique()); test_months=[m for m in months if m>=TEST_FROM]
A['a_init']=np.nan; A['a']=np.nan; gates_log=[]
for k in C:
    for mth in test_months:
        tr=A[(A.leg==k)&(A.month<mth)]; te=(A.leg==k)&(A.month==mth)
        if len(tr)<60 or te.sum()==0: continue
        G0=Gates(k,tr); A.loc[te,'a_init']=G0.act(A[te]); G=train(k,tr); A.loc[te,'a']=G.act(A[te])
        gates_log.append(dict(leg=k,month=mth,n_train=len(tr),**{f"mu_{c}":round(G.mu[i],3) for i,c in enumerate(G.f)},**{f"sig_{c}":round(float(np.exp(G.ls[i])),3) for i,c in enumerate(G.f)},**{f"w_{c}":round(G.w[i],2) for i,c in enumerate(G.f)},b=round(G.b[0],2),in_sample_roi=round(((G.act(tr)>=0.5)*tr.pnl).sum()/(STAKE*max((G.act(tr)>=0.5).sum(),1)),3)))
T=A[A.a.notna()].copy(); print(f"test months {T.month.min()}..{T.month.max()}, candidates: {T.leg.value_counts().to_dict()}  (hurdle ${HURDLE}/trade, reg {REG}, {EPOCHS} epochs)")
def hard(T):
    H=pd.concat([T[(T.leg==k)&T.price.between(*HARD[k])] for k in HARD]); yn=set(zip(H[H.side=='yes'].city,H[H.side=='yes'].mday))
    return pd.concat([H[H.side=='yes'],H[(H.side=="no")&np.array([k in yn for k in zip(H.city,H.mday)])]])
def soft(T,col='a',thr=0.5):
    S=T[T[col]>=thr]; yn=set(zip(S[S.side=='yes'].city,S[S.side=='yes'].mday)); return pd.concat([S[S.side=='yes'],S[(S.side=="no")&np.array([k in yn for k in zip(S.city,S.mday)])].sort_values(col,ascending=False).drop_duplicates(['city','mday'])])
H=hard(T); S0=soft(T,'a_init'); S=soft(T,'a')
print("\n### out-of-sample, same months")
print(pd.DataFrame({'hard rule':summ(H),'soft, untrained (init = hard bands)':summ(S0),'soft, TRAINED on P&L (a>=0.5)':summ(S),'soft, trained, sized by a (all candidates)':summ(T,T.a.values)}).T.round(2).to_string())
print("\nper leg (hard vs trained soft):"); print(pd.DataFrame({**{f"hard {k}":summ(H[H.leg==k]) for k in HARD},**{f"soft {k}":summ(S[S.leg==k]) for k in HARD}}).T.round(2).to_string())
bm=lambda x: x.groupby('month').pnl.sum().round(0); print("\nby month:"); print(pd.DataFrame({'hard':bm(H),'soft trained':bm(S)}).T.to_string())
print("by city:"); print(pd.DataFrame({'hard':H.groupby('city').pnl.sum(),'soft trained':S.groupby('city').pnl.sum()}).round(0).T.to_string())
kh=set(zip(H.city,H.mday,H.lo,H.side)); ks=set(zip(S.city,S.mday,S.lo,S.side)); ho=H[[k not in ks for k in zip(H.city,H.mday,H.lo,H.side)]]; so=S[[k not in kh for k in zip(S.city,S.mday,S.lo,S.side)]]
print(f"\noverlap: both {len(kh&ks)}  hard-only {len(kh-ks)}  soft-only {len(ks-kh)}"); print("hard-only:", summ(ho).round(2).to_dict()); print("soft-only:", summ(so).round(2).to_dict())
if len(so): print(so.groupby(['leg',pd.cut(so.price,[0,.1,.2,.3,.4,.5,.55,.6,.7])],observed=True).apply(summ)[['n','win','pnl','roi']].round(2).to_string())
if len(ho): print("hard-only by leg x price:"); print(ho.groupby(['leg',pd.cut(ho.price,[0,.1,.2,.3,.4,.5,.55,.6,.7])],observed=True).apply(summ)[['n','win','pnl','roi']].round(2).to_string())
print("\n### threshold on the trained intensity a (out-of-sample):"); print(pd.DataFrame({f"a>={t:.1f}":summ(soft(T,'a',t)) for t in [0.3,0.4,0.5,0.6,0.7,0.8]}).T.round(2).to_string())
GL=pd.DataFrame(gates_log); print("\n### trained gates, last refit per leg (centre / sigma / weight per feature; b = bias):")
print(GL.sort_values('month').groupby('leg').tail(1).set_index('leg').T.to_string())
print("\nin-sample ROI of the trained policy per refit (leg x month) — compare with the out-of-sample table above:"); print(GL.pivot(index='leg',columns='month',values='in_sample_roi').to_string())
T.drop(columns=[c for c in T.columns if str(T[c].dtype).startswith('category')]).to_parquet('out/soft3_candidates.parquet')
