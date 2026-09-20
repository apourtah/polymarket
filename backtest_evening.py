"""Backtest of the production evening bot (bot/evening_bot.py + bot/evening_config.py) over Jan 18 - Sep 16, 2026.
Same features, same Models class (EWMA per city + pooled ridge, refit nightly on all history to date), same rule
(agree -> agreed bucket if ask <= 0.60; disagree -> buckets with avg-P - ask >= 0.10 and ask <= 0.60), same sizing
($50, $100 when edge >= 0.20, $100/market, $200/city-day). Entry = bucket mid at 21:35 local the evening before
(bot start jitter median), taker fee 0.05*p*(1-p)/share. History/ridge pool = 6 cities as in the bot's history file;
trades only in the production CITIES. No decoys, no size jitter."""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, warnings, sys, os
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sweep_local import TZ
from bot import evening_config as C
warnings.filterwarnings('ignore'); pd.set_option('display.width',240)
for k in ('AGREE_MIN_PRICE','AGREE_MAX_PRICE','DISAGREE_MAX_PRICE','DISAGREE_EDGE'):
    if os.environ.get(k): setattr(C,k,float(os.environ[k]))
if os.environ.get('CITIES'): C.CITIES=os.environ['CITIES'].split(',')
MODES={}
if os.environ.get('MODES'):
    for tok in os.environ['MODES'].split(','):
        c,m=tok.split(':'); MODES[c]=set(m.split('+'))
    C.CITIES=list(MODES)
MODEL_MAX=float(os.environ.get('MODEL_MAX','0.60'))     # max price for the per-city model-bucket disagree modes
rh=lambda x: int(Decimal(str(x)).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
POOL=os.environ.get('POOL','Atlanta,Austin,Houston,Los Angeles,Miami,Seattle').split(','); TRADE=C.CITIES; site=json.load(open('out/stations.json'))
START=dt.date.fromisoformat(os.environ.get('START','2026-01-18')); END=dt.date.fromisoformat(os.environ.get('END','2026-09-16')); ENTRY_MIN=int(os.environ.get('ENTRY_MIN','35')); SLIP=float(os.environ.get('SLIP','0')); MIN_PX=float(os.environ.get('MIN_PX','0'))
# ---------- panel (same construction as correction_lab / bot history) ----------
h=pd.concat([pd.read_parquet(f) for f in ('data/hrrr_t2m.parquet','data/hrrr_t2m_east.parquet') if os.path.exists(f)]); h=h[(h.city!='ERR')&(h.run==0)].drop_duplicates(['date','run','fxx','city']).copy(); h['valid_utc']=pd.to_datetime(h.valid_utc,utc=True)
h['local']=[t.tz_convert(TZ[ct]) for t,ct in zip(h.valid_utc,h.city)]; h['mday']=[x.date() for x in h.local]; h['lh']=[x.hour for x in h.local]
cnt=h[h.lh.between(11,18)].groupby(['city','mday']).lh.nunique(); hm=h[h.lh.between(11,18)].groupby(['city','mday']).t2m_f.max(); hm=hm[cnt>=6]
met=pd.read_parquet('data/metar.parquet'); met['day']=met.local_time.dt.date; met['tf']=(met.temp_c*9/5+32).map(rh)
dmax=met.groupby(['station','day']).tf.max(); dmin=met.groupby(['station','day']).tf.min()
w=pd.read_parquet('data/hrrr_wx_d1.parquet'); w['mday']=w.time.dt.date; w['lh']=w.time.dt.hour; aft=w[w.lh.between(11,18)]
wx=aft.groupby(['city','mday']).agg(dew=('dew_point_2m','mean'),rhum=('relative_humidity_2m','mean'),wind=('wind_speed_10m','mean'),cloud=('cloud_cover','mean'),rad=('shortwave_radiation','mean'),pres=('surface_pressure','mean'),precip=('precipitation','sum'),
    wdir=('wind_direction_10m',lambda s: np.degrees(np.arctan2(np.sin(np.radians(s)).mean(),np.cos(np.radians(s)).mean()))%360))
morn=w[w.lh.between(4,8)].groupby(['city','mday']).agg(t_morn=('temperature_2m','mean'),cloud_morn=('cloud_cover','mean'))
mos=pd.read_parquet('data/mos_nbm.parquet'); mos['runtime']=pd.to_datetime(mos.runtime,utc=True); mos['ftime']=pd.to_datetime(mos.ftime,utc=True)
def mos_daily(model):
    d=mos[(mos.model==model)&(mos.runtime.dt.hour.isin([0,1]))].copy(); out=[]      # IEM stored NBS runs as 01Z before 2026
    for c,g in d.groupby('city'):
        tz=ZoneInfo(TZ[c]); g=g.copy(); lt=g.ftime.dt.tz_convert(tz); g['md']=lt.dt.date; g['lh']=lt.dt.hour; g=g[g.lh.between(11,18)]
        for md,gg in g.groupby('md'):
            cutoff=pd.Timestamp(dt.datetime(md.year,md.month,md.day,21,tzinfo=tz)-dt.timedelta(days=1)); gg=gg[gg.runtime<=cutoff]
            if gg.empty: continue
            last=gg[gg.runtime==gg.runtime.max()]; out.append(dict(city=c,mday=md,tmax=pd.to_numeric(last.tmp,errors='coerce').max()))
    return pd.DataFrame(out).set_index(['city','mday'])
nbs=mos_daily('NBS').rename(columns={'tmax':'nbm_max'}); gfs=mos_daily('GFS').rename(columns={'tmax':'gfsmos_max'})
rows=[]
for c in POOL:
    st=site[c][0]
    for (cc,d),v in hm.items():
        if cc!=c: continue
        act=dmax.get((st,d),np.nan)
        if pd.isna(act): continue
        r=dict(city=c,mday=d,hrrr=v,actual=act,err=act-v,yday_max=dmax.get((st,d-dt.timedelta(days=1)),np.nan),yday_min=dmin.get((st,d-dt.timedelta(days=1)),np.nan),doy=d.timetuple().tm_yday)
        for src in (wx,morn,nbs,gfs):
            if (c,d) in src.index: r.update(src.loc[(c,d)].to_dict())
        rows.append(r)
P=pd.DataFrame(rows).sort_values(['city','mday']).reset_index(drop=True)
P['yday_err']=P.groupby('city').err.shift(1); P['e2']=P.groupby('city').err.shift(2)
P['r5']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(5,min_periods=3).mean()); P['r14']=P.groupby('city').err.transform(lambda s: s.shift(1).rolling(14,min_periods=5).mean())
P['dpd']=P.hrrr-P.dew; P['wdir_s']=np.sin(np.radians(P.wdir)); P['wdir_c']=np.cos(np.radians(P.wdir)); P['nbm_minus_hrrr']=P.nbm_max-P.hrrr; P['gfs_minus_hrrr']=P.gfsmos_max-P.hrrr
P.to_parquet('out/backtest_panel.parquet'); print(f"panel: {len(P)} city-days {P.mday.min()}..{P.mday.max()}, NBM {P.nbm_max.notna().mean():.0%}, GFS-MOS {P.gfsmos_max.notna().mean():.0%}")
# ---------- the bot's Models (copied verbatim from bot/evening_bot.py) ----------
class Models:
    def __init__(self, hist):
        self.hist = hist.dropna(subset=["err"]).copy(); self.cities = sorted(self.hist.city.unique())
        self.ewma = {}; self.ewma_sd = {}; self.ridge_sd = {}
        for c in self.cities:
            e = self.hist[self.hist.city == c].sort_values("mday").err.values
            best = None
            for k in C.EWMA_GAINS:
                b = 0.0; se = []
                for x in e: se.append(x - b); b += k * (x - b)
                s = float(np.sum(np.square(se)))
                if best is None or s < best[0]: best = (s, k, b, np.std(se[-60:]))
            self.ewma[c] = (best[1], best[2]); self.ewma_sd[c] = max(best[3], 1.0)
        X = self.hist[C.FEATS].copy(); self.med = X.median().fillna(0); X = X.fillna(self.med); self.mu = X.mean(); self.sd = X.std().replace(0, 1)
        Z = np.c_[((X - self.mu) / self.sd).values, pd.get_dummies(self.hist.city).reindex(columns=self.cities, fill_value=0).values]
        self.ridge = Ridge(alpha=C.RIDGE_ALPHA).fit(Z, self.hist.err.values)
        res = self.hist.err.values - self.ridge.predict(Z)
        for c in self.cities: self.ridge_sd[c] = max(float(np.std(res[(self.hist.city == c).values][-60:])), 1.0)
    def predict(self, city, feats):
        x = pd.Series({f: feats.get(f, np.nan) for f in C.FEATS}).fillna(self.med)
        z = np.r_[((x - self.mu) / self.sd).values, [1.0 if c == city else 0.0 for c in self.cities]]
        return dict(ewma=self.ewma[city][1], ridge=float(self.ridge.predict(z[None, :])[0]), ewma_sd=self.ewma_sd[city], ridge_sd=self.ridge_sd[city])
def bucket_probs(mu, sd, buckets):
    return {b: norm.cdf((min(b[1], 200) + 0.5 - mu) / sd) - norm.cdf((max(b[0], -200) - 0.5 - mu) / sd) for b in buckets}
# ---------- markets & prices ----------
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
ms=[m for m in json.load(open('data/markets.json')) if re.search(rf'highest temperature in ({"|".join(TRADE)}) be ', m['question'] or '') and (m.get('closed') or os.environ.get('INCLUDE_OPEN'))]   # INCLUDE_OPEN=1: also unresolved markets (won/pnl meaningless for them; see 'closed' in the bucket dump)
mk={}
for m in ms:
    city=re.search(r'temperature in (.+?) be ',m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10])
    mk.setdefault((city,d),{})[bucket(m['question'])]=m
def yes_price(m, ts):
    try: h=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: return None
    hs=h['history']
    if not hs: return None
    i=bisect.bisect_right([x['t'] for x in hs],ts)-1
    if i<0 or ts-hs[i]['t']>2*3600: return None
    p=hs[i]['p']; return p if h['losing_outcome']=='Yes' else 1-p
# ---------- walk-forward, nightly refit like the bot ----------
trades=[]; nights=[]; allb=[]   # allb: every priced bucket per night (for legs_backtest.py)
for d in sorted(P[(P.mday>=START)&(P.mday<=END)].mday.unique()):
    H=P[P.mday<d].dropna(subset=['err'])
    if len(H)<C.MIN_HISTORY_DAYS: continue
    M=Models(H)
    for c in TRADE:
        r=P[(P.city==c)&(P.mday==d)]
        if r.empty or (c,d) not in mk: continue
        r=r.iloc[0]; feats=r.to_dict(); p=M.predict(c,feats); mu_e,mu_r=r.hrrr+p['ewma'],r.hrrr+p['ridge']
        bk=mk[(c,d)]; tz=ZoneInfo(TZ[c]); ts=int(dt.datetime(d.year,d.month,d.day,21,tzinfo=tz).timestamp())-86400+ENTRY_MIN*60
        prices={b:yes_price(m,ts) for b,m in bk.items()}; prices={b:v for b,v in prices.items() if v is not None}
        if len(prices)<3: continue
        buckets=list(prices); Pe=bucket_probs(mu_e,p['ewma_sd'],buckets); Pr=bucket_probs(mu_r,p['ridge_sd'],buckets)
        be,br=max(Pe,key=Pe.get),max(Pr,key=Pr.get); agree=be==br
        nights.append(dict(city=c,mday=d,agree=agree,win_lo=r.actual-r.actual%2,be=be[0],br=br[0],p_be=prices[be],p_br=prices[br],pe_be=Pe[be],pr_br=Pr[br],pav_be=(Pe[be]+Pr[be])/2,pav_br=(Pe[br]+Pr[br])/2))
        fav=max(prices,key=prices.get)
        for b in buckets:
            allb.append(dict(city=c,mday=d,lo=b[0],hi=b[1],price=prices[b],pe=Pe[b],pr=Pr[b],won=[float(x) for x in bk[b]['outcome_prices']]==[1.0,0.0],closed=bool(bk[b].get('closed')),agree=agree,be=be[0],br=br[0],fav=fav[0],fav_p=prices[fav],win_lo=r.actual-r.actual%2))
        spent=0.0
        for b in buckets:
            ask=prices[b]+SLIP; pav=(Pe[b]+Pr[b])/2; why=None
            if prices[b]<MIN_PX: continue
            modes=MODES.get(c,{'agree','edge'})
            if agree and 'agree' in modes and b==be and C.AGREE_MIN_PRICE<=ask<=C.AGREE_MAX_PRICE: why='agree'
            elif (not agree) and 'edge' in modes and pav-ask>=C.DISAGREE_EDGE and ask<=C.DISAGREE_MAX_PRICE: why='disagree'
            elif (not agree) and 'ridge' in modes and b==br and ask<=MODEL_MAX: why='dis_ridge'
            elif (not agree) and 'ewma' in modes and b==be and ask<=MODEL_MAX: why='dis_ewma'
            if not why: continue
            edge=pav-ask; stake=C.STAKE*(2 if (why=='disagree' and edge>=2*C.DISAGREE_EDGE) else 1); stake=min(stake,C.MAX_PER_MARKET_USD,C.MAX_PER_CITY_DAY_USD-spent)
            if stake<C.MIN_ORDER_SHARES*ask: continue
            spent+=stake; sh=stake/ask; won=[float(x) for x in bk[b]['outcome_prices']]==[1.0,0.0]; fee=0.05*ask*(1-ask)*sh
            trades.append(dict(city=c,mday=d,bucket=b,why=why,price=ask,P=pav,edge=edge,stake=stake,won=won,pnl=(sh-stake if won else -stake)-fee))
T=pd.DataFrame(trades); N=pd.DataFrame(nights); T.to_parquet('out/backtest_evening_trades.parquet'); N.to_parquet('out/backtest_evening_nights.parquet'); pd.DataFrame(allb).to_parquet('out/backtest_evening_buckets.parquet')
def summ(t): return pd.Series(dict(n=len(t),win=t.won.mean(),avg_px=t.price.mean(),stake=t.stake.sum(),pnl=t.pnl.sum(),roi=t.pnl.sum()/t.stake.sum() if len(t) else np.nan))
print(f"\n=== production rule, {START}..{END}, cities {TRADE}, entry mid at 21:{ENTRY_MIN:02d} local evening before, slip {SLIP:.2f}")
print(f"nights evaluated: {len(N)}  agree share {N.agree.mean():.0%}  EWMA-bucket hit {(N.be==N.win_lo).mean():.3f}  ridge hit {(N.br==N.win_lo).mean():.3f}")
print("\nby signal:"); print(T.groupby('why').apply(summ).round(3).to_string())
print("\nby city:"); print(T.groupby('city').apply(summ).round(3).to_string())
T['month']=pd.to_datetime(T.mday).dt.to_period('M'); print("\nby month:"); print(T.groupby('month').apply(summ).round(2).to_string())
print("\nALL:", summ(T).round(3).to_dict())
# capital & drawdown
daily=T.groupby('mday').agg(stake=('stake','sum'),pnl=('pnl','sum')); cum=daily.pnl.cumsum(); dd=(cum-cum.cummax()).min()
print(f"days with trades {len(daily)}, max $ deployed in a day {daily.stake.max():.0f}, mean {daily.stake.mean():.0f}; max drawdown ${dd:.0f}; final ${cum.iloc[-1]:.0f}")
T['bin']=pd.cut(T.price,[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6]); print("\nby entry price bin:"); print(T.groupby('bin',observed=True).apply(summ).round(3).to_string())
dis=T[T.why=='disagree']; top=dis.sort_values('pnl',ascending=False).head(5); print("\ntop 5 disagree wins:"); print(top[['city','mday','bucket','price','P','stake','pnl']].to_string())
print(f"disagree leg without its best 3 trades: ${dis.pnl.sum()-dis.pnl.nlargest(3).sum():.0f} on ${dis.stake.sum():.0f}")
N['month']=pd.to_datetime(N.mday).dt.to_period('M'); print("\nbucket hit rate by month (EWMA / ridge / market favorite n/a):"); print(N.groupby('month').apply(lambda g: pd.Series(dict(n=len(g),agree=g.agree.mean(),ewma_hit=(g.be==g.win_lo).mean(),ridge_hit=(g.br==g.win_lo).mean()))).round(2).to_string())
