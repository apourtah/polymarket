"""Exit-strategy sweep on the production backtest trades (4 cities, current rule, realistic entry):
for each trade follow the YES 5-min mid from entry to resolution; test take-profit rules (multiple of entry,
absolute level, with/without stop-loss, full or half exit) per entry-price band. Sell = mid - 1c, taker fee."""
import pandas as pd, numpy as np, json, re, bisect, datetime as dt, sys
from zoneinfo import ZoneInfo
from sweep_local import TZ
pd.set_option('display.width',240)
T=pd.read_parquet(sys.argv[1] if len(sys.argv)>1 else 'out/bt_trades_4city_realistic.parquet')
def bucket(q):
    mm=re.search(r'between (-?\d+)-(-?\d+)|(-?\d+)°[FC] or below|(-?\d+)°[FC] or (?:above|higher)|(-?\d+)°[FC] on',q)
    if mm.group(1): return (int(mm.group(1)),int(mm.group(2)))
    if mm.group(3): return (-999,int(mm.group(3)))
    if mm.group(4): return (int(mm.group(4)),999)
    return (int(mm.group(5)),int(mm.group(5)))
cities=sorted(T.city.unique())
ms=[m for m in json.load(open('data/markets.json')) if re.search(rf'highest temperature in ({"|".join(cities)}) be ', m['question'] or '') and m.get('closed')]
mk={}
for m in ms:
    city=re.search(r'temperature in (.+?) be ',m['question']).group(1); d=dt.date.fromisoformat(m['end_date'][:10]); mk[(city,d,bucket(m['question']))]=m
def yes_path(m, ts0):
    h=json.load(open(f"data/prices/{m['market_id']}.json")); hs=h['history']
    flip=h['losing_outcome']!='Yes'
    return [(x['t'],(1-x['p']) if flip else x['p']) for x in hs if x['t']>ts0]
paths=[]
for i,r in T.iterrows():
    m=mk.get((r.city,r.mday,tuple(r.bucket)))
    if m is None: continue
    tz=ZoneInfo(TZ[r.city]); d=r.mday; ts0=int(dt.datetime(d.year,d.month,d.day,21,tzinfo=tz).timestamp())-86400+35*60
    p=yes_path(m,ts0); paths.append(dict(idx=i,path=p,tz=tz,day=d))
print(f"{len(paths)} trades with price paths ({len(T)} trades)")
FEE=lambda p,sh: 0.05*p*(1-p)*sh
def simulate(rule, band):
    out=[]
    for pp in paths:
        r=T.loc[pp['idx']]
        if not band(r.price): continue
        sh=r.stake/r.price; buy_fee=FEE(r.price,sh); hold=(sh-r.stake if r.won else -r.stake)-buy_fee
        tp=rule.get('mult',None); lvl=rule.get('level',None); stop=rule.get('stop',None); frac=rule.get('frac',1.0); cutoff_h=rule.get('cutoff',None)
        target=min(r.price*tp,0.97) if tp else lvl
        exit_px=None; exit_t=None
        for t,p in pp['path']:
            lt=dt.datetime.fromtimestamp(t,pp['tz'])
            if cutoff_h is not None and lt.date()==pp['day'] and lt.hour>=cutoff_h: break
            if p>=target: exit_px=p; exit_t=lt; break
            if stop is not None and p<=r.price*stop: exit_px=p; exit_t=lt; break
        if exit_px is None: out.append(dict(pnl=hold,exited=False,won=r.won,stake=r.stake)); continue
        sell=max(exit_px-0.01,0.001); sh_x=sh*frac; sh_h=sh-sh_x
        pnl=sh_x*sell-FEE(sell,sh_x)+(sh_h if r.won else 0.0)-r.stake-buy_fee
        out.append(dict(pnl=pnl,exited=True,won=r.won,stake=r.stake,exit_px=exit_px,hour=exit_t.hour,same_day=exit_t.date()==pp['day']))
    o=pd.DataFrame(out)
    if o.empty: return None
    ex=o[o.exited]
    return dict(n=len(o),stake=o.stake.sum(),hold_pnl=None,pnl=round(o.pnl.sum()),roi=round(o.pnl.sum()/o.stake.sum(),3),exits=len(ex),exit_rate=round(len(ex)/len(o),2),
                exit_would_win=round(ex.won.mean(),2) if len(ex) else None,exit_would_lose=int((~ex.won).sum()) if len(ex) else 0,avg_exit_px=round(ex.exit_px.mean(),2) if len(ex) else None)
bands={'agree 35-60c':lambda p:0.35<=p<=0.61,'disagree <=7.5c':lambda p:p<0.10,'all':lambda p:True}
for bname,band in bands.items():
    print(f"\n==================== {bname} ====================")
    base=simulate({'level':9.9},band); print(f"HOLD to resolution: n {base['n']}, staked ${base['stake']:.0f}, P&L ${base['pnl']}, ROI {base['roi']:+.1%}")
    rows=[]
    for tp in [1.5,2,3,4,6,10]: rows.append(dict(rule=f'TP x{tp}',**simulate({'mult':tp},band)))
    for lvl in [0.5,0.6,0.7,0.8,0.9,0.95]: rows.append(dict(rule=f'TP @{lvl:.2f}',**simulate({'level':lvl},band)))
    for lvl in [0.7,0.8,0.9]: rows.append(dict(rule=f'TP @{lvl:.2f} half',**simulate({'level':lvl,'frac':0.5},band)))
    for lvl,stop in [(0.8,0.5),(0.9,0.5),(0.8,0.3)]: rows.append(dict(rule=f'TP @{lvl:.2f} SL x{stop}',**simulate({'level':lvl,'stop':stop},band)))
    for stop in [0.3,0.5]: rows.append(dict(rule=f'SL x{stop} only',**simulate({'level':9.9,'stop':stop},band)))
    R=pd.DataFrame(rows).drop(columns=['hold_pnl','n','stake']); print(R.to_string(index=False))
