"""Realised P&L per wallet per market from data-api /closed-positions (covers maker fills and early exits, unlike the taker tape),
for every wallet seen in the taker tapes with >= MIN_FILLS fills. Sorted by timestamp desc, stops paging once past FROM.
-> data/realized/<wallet>.json ; env FROM (2026-08-07) MIN_FILLS (3)"""
import json, os, time, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
FROM=os.environ.get('FROM','2026-08-07'); MIN_FILLS=int(os.environ.get('MIN_FILLS','3')); os.makedirs('data/realized',exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
T=pd.read_parquet('out/wallet_fills.parquet'); ws=T.groupby('w').size(); ws=ws[ws>=MIN_FILLS].index.tolist(); print(len(ws),"wallets",flush=True)
def one(w):
    path=f"data/realized/{w}.json"
    if os.path.exists(path): return 0
    rows=[]; off=0
    while True:
        for k in range(4):
            try: r=S.get("https://data-api.polymarket.com/closed-positions",params={"user":w,"limit":50,"offset":off,"sortBy":"TIMESTAMP","sortDirection":"DESC"},timeout=60)
            except Exception: time.sleep(2); continue
            if r.status_code==200: break
            time.sleep(1.5*(k+1))
        else: return -1
        j=r.json(); rows+=[{k:x.get(k) for k in ('conditionId','title','outcome','totalBought','avgPrice','realizedPnl','endDate','timestamp')} for x in j]
        if len(j)<50 or off>=3000 or (j and j[-1].get('endDate') and j[-1]['endDate']<FROM): break
        off+=50; time.sleep(0.1)
    json.dump(rows,open(path+'.tmp','w')); os.replace(path+'.tmp',path); time.sleep(0.1); return len(rows)
n=0
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(one,ws):
        n+=1
        if n%500==0: print(n,flush=True)
print("done",flush=True)
