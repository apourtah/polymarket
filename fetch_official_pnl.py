"""Official P&L curve per wallet (user-pnl-api, all markets, daily) for every tape wallet with >= MIN_FILLS fills.
-> data/pnl_curves/<wallet>.json"""
import json, os, time, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
MIN_FILLS=int(os.environ.get('MIN_FILLS','3')); os.makedirs('data/pnl_curves',exist_ok=True); S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
T=pd.read_parquet('out/wallet_fills.parquet'); ws=T.groupby('w').size(); ws=ws[ws>=MIN_FILLS].index.tolist(); print(len(ws),flush=True)
def one(w):
    p=f"data/pnl_curves/{w}.json"
    if os.path.exists(p): return 0
    for k in range(4):
        try:
            r=S.get("https://user-pnl-api.polymarket.com/user-pnl",params={"user_address":w,"interval":"all","fidelity":"1d"},timeout=60)
            if r.status_code==200: json.dump(r.json(),open(p,'w')); time.sleep(0.1); return 1
        except Exception: pass
        time.sleep(1.5*(k+1))
    return -1
n=0
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(one,ws):
        n+=1
        if n%500==0: print(n,flush=True)
print("done",flush=True)
