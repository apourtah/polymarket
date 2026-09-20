"""Taker-side trade tapes WITH wallets for the highest-temperature markets of the 7 production cities (data-api /trades,
one row per taker fill: proxyWallet, side, outcome, price, size, timestamp, transactionHash). -> data/tapes_w/<conditionId>.json
env: FROM (2026-08-07) TO (2026-09-18) CITIES"""
import json, os, re, time, requests, datetime as dt
from concurrent.futures import ThreadPoolExecutor
FROM=os.environ.get('FROM','2026-08-07'); TO=os.environ.get('TO','2026-09-18'); CITIES=os.environ.get('CITIES','Los Angeles,Austin,Chicago,Houston,Dallas,Seattle,Miami').split(',')
os.makedirs('data/tapes_w',exist_ok=True); S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
ms=[m for m in json.load(open('data/markets.json')) if re.search(rf'highest temperature in ({"|".join(CITIES)}) be ',m['question'] or '') and FROM<=m['end_date'][:10]<=TO]
print(len(ms),"markets",flush=True); KEEP=('proxyWallet','side','asset','outcome','price','size','timestamp','transactionHash','name','pseudonym')
def one(m):
    path=f"data/tapes_w/{m['condition_id']}.json"
    if os.path.exists(path): return 0
    rows=[]; off=0
    while True:
        for k in range(4):
            try: page=S.get("https://data-api.polymarket.com/trades",params={"market":m['condition_id'],"limit":1000,"offset":off},timeout=60); 
            except Exception: time.sleep(2); continue
            if page.status_code==200: break
            time.sleep(1.5*(k+1))
        else: return -1
        j=page.json(); rows+=[{k:r.get(k) for k in KEEP} for r in j]
        if len(j)<1000 or off>=20000: break
        off+=1000; time.sleep(0.12)
    json.dump(dict(market_id=m['market_id'],question=m['question'],end_date=m['end_date'],outcome_prices=m['outcome_prices'],tokens=m['tokens'],trades=rows),open(path+'.tmp','w')); os.replace(path+'.tmp',path); time.sleep(0.12); return len(rows)
n=0
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(one,ms):
        n+=1
        if n%200==0: print(n,flush=True)
print("done",flush=True)
