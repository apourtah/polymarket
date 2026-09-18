"""YES-only sweep: trigger when YES midpoint first touches TRIG; rest a bid at BID; fill on the
first taker print that would hit it (SELL YES <= BID or BUY NO >= 1-BID). Maker, no fee, hold to
resolution. Grid over TRIG x BID. Also splits fills by local time of the dip."""
import json, glob, os, bisect, datetime as dt, numpy as np, pandas as pd
from zoneinfo import ZoneInfo
from sweep_local import TZ, CITY_RE
TRIGS=[0.97,0.98,0.99,0.995]; BIDS=[0.90,0.93,0.95,0.96,0.97,0.98,0.99]; STAKE=50.0
mk={m['market_id']:m for m in json.load(open('data/markets.json'))}; by_cid={m['condition_id']:m for m in mk.values()}
rows=[]
for f in glob.glob('data/tapes/*.json'):
    cid=os.path.basename(f)[:-5]; m=by_cid.get(cid)
    if not m: continue
    try: ph=json.load(open(f"data/prices/{m['market_id']}.json"))
    except FileNotFoundError: continue
    if not ph['history']: continue
    hs=sorted((x['t'],x['p']) for x in ph['history']); ts=[t for t,_ in hs]
    losing=ph['losing_outcome']; yes=[p if losing=='Yes' else 1-p for _,p in hs]; won=(losing=='No')
    ytok=str(m['tokens'][m['outcomes'].index('Yes')]); ntok=str(m['tokens'][m['outcomes'].index('No')])
    tape=json.load(open(f))
    sells=sorted([(x['timestamp'],float(x['price']),float(x['size'])) for x in tape if str(x['asset'])==ytok and x['side']=='SELL']+
                 [(x['timestamp'],1-float(x['price']),float(x['size'])) for x in tape if str(x['asset'])==ntok and x['side']=='BUY'])
    cm=CITY_RE.match(m['question'] or ''); tz=ZoneInfo(TZ[cm.group(1)]) if cm and cm.group(1) in TZ else None
    for trig in TRIGS:
        i0=next((i for i,p in enumerate(yes) if p>=trig),None)
        if i0 is None: continue
        t0=ts[i0]; after=[s for s in sells if s[0]>t0]
        for bid in BIDS:
            if bid>=trig: continue
            h=next((s for s in after if s[1]<=bid),None)
            rec=dict(market_id=m['market_id'],trig=trig,bid=bid,won=won,filled=h is not None)
            if h:
                t,p,s=h; sh=min(STAKE/bid,s); rec.update(sh=sh,usd=sh*bid,pnl=sh*(1.0 if won else 0.0)-sh*bid,
                    lhour=dt.datetime.fromtimestamp(t,tz).hour if tz else None, tfill=t, tclose=m['closed_time'], unit=('F' if '°F' in m['question'] else 'C'))
            rows.append(rec)
df=pd.DataFrame(rows); df.to_parquet('out/dip_sweep.parquet'); print(len(df),'rows')
