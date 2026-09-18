"""Resting-bid-on-the-dip backtest.
For every token (YES and NO of each market with a cached tape): t0 = first 5-min midpoint >= TRIG.
After t0, a resting bid at BID fills at the first taker print that would hit it: a taker SELL of
our token at price <= BID, or a taker BUY of the complement at price >= 1-BID. Fill size = min of
our $STAKE and the print. Maker: no fee. Hold to resolution.
"""
import json, glob, os, sys, bisect, numpy as np, pandas as pd
TRIG=0.995; STAKE=50.0
BIDS=[0.95,0.97,0.98]
mk={m['market_id']:m for m in json.load(open('data/markets.json'))}
by_cid={m['condition_id']:m for m in mk.values()}
rows=[]
for f in glob.glob('data/tapes/*.json'):
    cid=os.path.basename(f)[:-5]; m=by_cid.get(cid)
    if not m: continue
    try: ph=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: continue
    if not ph['history']: continue
    hs=sorted((x['t'],x['p']) for x in ph['history']); ts=[t for t,_ in hs]
    losing=ph['losing_outcome']; tape=json.load(open(f))
    for side_name in ('Yes','No'):
        tok=str(m['tokens'][m['outcomes'].index(side_name)]); other=str(m['tokens'][1-m['outcomes'].index(side_name)])
        ser=[p if losing==side_name else 1-p for _,p in hs]          # this token's midpoint series
        won=(losing!=side_name)
        i0=next((i for i,p in enumerate(ser) if p>=TRIG),None)
        if i0 is None: continue
        t0=ts[i0]
        # prints after t0 that would hit a resting bid on this token
        hits=[]
        for x in tape:
            if x['timestamp']<=t0: continue
            p=float(x['price']); s=float(x['size'])
            if str(x['asset'])==tok and x['side']=='SELL': hits.append((x['timestamp'],p,s))
            elif str(x['asset'])==other and x['side']=='BUY': hits.append((x['timestamp'],1-p,s))
        hits.sort()
        rec=dict(market_id=m['market_id'],cid=cid,token=side_name,won=won,t0=t0,question=m['question'],n_prints_after=len(hits),
                 min_print_after=min([h[1] for h in hits],default=np.nan),
                 mid_min_after=min(ser[i0:]),)
        for b in BIDS:
            h=[x for x in hits if x[1]<=b]
            if h:
                t,p,s=h[0]; sh=min(STAKE/b,s); rec[f'fill{b}']=True; rec[f'sh{b}']=sh; rec[f'pnl{b}']=(sh*(1.0 if won else 0.0)-sh*b); rec[f'tfill{b}']=t
                # did the midpoint recover to >= TRIG after the fill?
                j=bisect.bisect_left(ts,t); rec[f'recover{b}']=any(x>=TRIG for x in ser[j:])
            else: rec[f'fill{b}']=False; rec[f'pnl{b}']=0.0; rec[f'sh{b}']=0.0
        rows.append(rec)
df=pd.DataFrame(rows); df.to_parquet('out/dip_study.parquet')
pd.set_option('display.width',220)
print(f"{len(df)} tokens reached {TRIG} (from {df.market_id.nunique()} markets with tapes); token later LOST in {(~df.won).mean():.2%}")
for b in BIDS:
    f=df[df[f'fill{b}']]
    print(f"\nresting bid @ {b}: filled {len(f)} ({len(f)/len(df):.1%} of tokens) | avg fill ${f[f'sh{b}'].mean()*b:.0f} | token lost {(~f.won).mean():.2%} (breakeven {(1-b)/1:.2%} loss rate) | "
          f"P&L ${f[f'pnl{b}'].sum():,.0f} on ${(f[f'sh{b}']*b).sum():,.0f} ({f[f'pnl{b}'].sum()/(f[f'sh{b}']*b).sum():+.2%}) | mid recovered to >= {TRIG} after fill: {f[f'recover{b}'].mean():.1%}")
    print("   by token side:", f.groupby('token').apply(lambda g: f"n={len(g)} lost={(~g.won).mean():.2%} roi={g[f'pnl{b}'].sum()/(g[f'sh{b}']*b).sum():+.2%}").to_dict())
