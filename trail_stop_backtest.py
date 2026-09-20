"""Current strategy (agree 10-53c / model picks 5-45c / NO leg H 35-55c) with a TRAILING stop, on the 5-min mid paths.
For each trade: path of our position's price from entry (21:35 local eve-before +1c) to resolution; trail = exit when the price
falls TRAIL below its running max since entry (absolute cents); optional activation only after the position is up ACT cents;
exit at mid - 1c with taker fee. Compared with hold-to-resolution. Uses out/backtest_evening_buckets.parquet + data/prices."""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, os, sys
from zoneinfo import ZoneInfo
from sweep_local import TZ
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
STAKE=50.0; SLIP=0.01
B=pd.read_parquet('out/backtest_evening_buckets.parquet'); B['mday']=pd.to_datetime(B.mday); B=B[B.closed] if 'closed' in B else B
MODES={"Los Angeles":{"agree","ewma"},"Austin":{"agree","ewma"},"Chicago":{"agree","ridge"},"Houston":{"ridge"},"Dallas":{"ridge"},"Seattle":{"agree"},"Miami":{"agree"}}
B['is_be']=B.lo==B.be; B['is_br']=B.lo==B.br; B['is_fav']=B.lo==B.fav; B['modes']=B.city.map(lambda c: ','.join(sorted(MODES[c])))
Y=pd.concat([B[B.agree&B.is_be&B.modes.str.contains('agree')&B.price.between(0.10,0.53)].assign(leg='agree'),B[~B.agree&B.is_br&B.modes.str.contains('ridge')&B.price.between(0.05,0.45)].assign(leg='dis_ridge'),B[~B.agree&B.is_be&B.modes.str.contains('ewma')&B.price.between(0.05,0.45)].assign(leg='dis_ewma')]).assign(side='yes')
yk=Y.set_index(['city','mday']); B['yes_lo']=[yk.lo.get((c,d),np.nan) for c,d in zip(B.city,B.mday)]; B['yleg']=[yk.leg.get((c,d)) for c,d in zip(B.city,B.mday)]
dis=B[B.yleg.isin(['dis_ridge','dis_ewma'])]; N=pd.concat([dis[dis.is_fav&(dis.lo>dis.yes_lo)],dis[(dis.yleg=='dis_ridge')&(dis.lo==dis.br+2)]]).drop_duplicates(['city','mday','lo']); N=N[N.price.between(0.35,0.55)].assign(leg='no_H',side='no')
T=pd.concat([Y,N]).reset_index(drop=True); print(f"{len(T)} trades ({(T.side=='yes').sum()} YES, {(T.side=='no').sum()} NO), {T.mday.min().date()}..{T.mday.max().date()}")
# market ids
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
mid={}
for m in json.load(open('data/markets.json')):
    q=m.get('question') or ''; mm=re.search(r'highest temperature in (.+?) be ',q)
    if mm and m.get('closed') and m.get('end_date'): mid[(mm.group(1),m['end_date'][:10],bucket(q)[0])]=(m['market_id'],m.get('tokens'))
T['market_id']=[mid.get((c,d.strftime('%Y-%m-%d'),int(lo)),(None,None))[0] for c,d,lo in zip(T.city,T.mday,T.lo)]; print("with price file:", T.market_id.notna().sum())
TRAILS=[0.05,0.08,0.10,0.15,0.20,0.30]; ACTS=[0.0,0.05,0.10,0.20]
res={(t,a):[] for t in TRAILS for a in ACTS}; hold=[]; paths_ok=0
for r in T.itertuples():
    if r.market_id is None: continue
    try: h=json.load(open(f"data/prices/{r.market_id}.json"))
    except FileNotFoundError: continue
    hs=h['history']
    if not hs: continue
    ts=np.array([x['t'] for x in hs]); p=np.array([x['p'] for x in hs]); yes=p if h['losing_outcome']=='Yes' else 1-p
    tz=ZoneInfo(TZ[r.city]); t0=int(dt.datetime(r.mday.year,r.mday.month,r.mday.day,21,35,tzinfo=tz).timestamp())-86400
    i0=bisect.bisect_right(ts,t0); pos=yes[i0:] if r.side=='yes' else 1-yes[i0:]      # our position's price path after entry
    if len(pos)<2: continue
    paths_ok+=1; entry=(r.price if r.side=='yes' else 1-r.price)+SLIP; sh=STAKE/entry; fee_in=0.05*entry*(1-entry)*sh
    won=r.won if r.side=='yes' else (not r.won); final=1.0 if won else 0.0; pnl_hold=sh*final-STAKE-fee_in; hold.append(pnl_hold)
    runmax=np.maximum.accumulate(np.maximum(pos,entry))
    for t in TRAILS:
        for a in ACTS:
            act=(runmax>=entry+a); trig=act&(pos<=runmax-t); k=np.argmax(trig) if trig.any() else -1
            if k>=0 and pos[k]>0.01 and pos[k]<0.995:
                px=max(pos[k]-SLIP,0.005); fee_out=0.05*px*(1-px)*sh; pnl=sh*px-STAKE-fee_in-fee_out; res[(t,a)].append((pnl,1,r.leg,r.month if hasattr(r,'month') else str(r.mday)[:7]))
            else: res[(t,a)].append((pnl_hold,0,r.leg,str(r.mday)[:7]))
H=np.array(hold); print(f"\npaths available: {paths_ok}; HOLD to resolution: P&L ${H.sum():+,.0f} on ${STAKE*len(H):,.0f} ({H.sum()/(STAKE*len(H)):+.1%}), max drawdown of the daily curve computed below")
rows=[]
for (t,a),v in res.items():
    d=pd.DataFrame(v,columns=['pnl','stopped','leg','month']); m=d.groupby('month').pnl.sum()
    rows.append(dict(trail=t,activate_after=a,n=len(d),stopped=d.stopped.mean(),pnl=d.pnl.sum(),roi=d.pnl.sum()/(STAKE*len(d)),vs_hold=d.pnl.sum()-H.sum(),neg_months=(m<0).sum(),worst_month=m.min(),yes_pnl=d[d.leg!='no_H'].pnl.sum(),no_pnl=d[d.leg=='no_H'].pnl.sum()))
R=pd.DataFrame(rows).sort_values(['activate_after','trail']); print("\n### trailing stop grid (trail = cents below the running max; activate_after = only once the position is up this much)"); print(R.round(2).to_string(index=False))
# what the stops do: distribution of exits for the best-looking cell and for a tight one
for key in [(0.10,0.0),(0.20,0.10)]:
    d=pd.DataFrame(res[key],columns=['pnl','stopped','leg','month']); print(f"\ntrail {key[0]:.2f} / activate {key[1]:.2f}: stopped {d.stopped.mean():.0%} of trades; P&L of stopped trades ${d[d.stopped==1].pnl.sum():+,.0f} vs what holding them would have made ${H[d.stopped.values==1].sum():+,.0f}")
