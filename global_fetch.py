"""Global cities: markets + YES price histories (10-min) for the highest-temperature events of the 37 non-US cities,
FROM..TO, via Gamma event slug + CLOB prices-history. -> data/global/markets.json, data/global/prices/<market_id>.json
env FROM (2026-06-15) TO (2026-09-18)"""
import json, os, re, time, requests, datetime as dt, pandas as pd
from concurrent.futures import ThreadPoolExecutor
FROM=os.environ.get('FROM','2026-06-15'); TO=os.environ.get('TO','2026-09-18'); os.makedirs('data/global/prices',exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
site=json.load(open('out/stations.json')); US={'Chicago','Denver','New York City','Dallas','Atlanta','Miami','Austin','Houston','Seattle','Los Angeles','San Francisco'}
CITIES=[c for c in site if c not in US]; SLUG={c:re.sub(r'[^a-z0-9]+','-',c.lower()).strip('-') for c in CITIES}; SLUG['Seoul (Incheon)']='seoul'
mpath='data/global/markets.json'; ms=json.load(open(mpath)) if os.path.exists(mpath) else []; have={m['market_id'] for m in ms}
def get(url,params):
    for k in range(4):
        try:
            r=S.get(url,params=params,timeout=60)
            if r.status_code==200: return r.json()
        except Exception: pass
        time.sleep(1.5*(k+1))
    return None
def event(c,d):
    slug=f"highest-temperature-in-{SLUG[c]}-on-{d.strftime('%B').lower()}-{d.day}-{d.year}"; ev=get("https://gamma-api.polymarket.com/events",{"slug":slug})
    if not ev: return []
    e=ev[0]; out=[]
    for mk in e.get('markets',[]):
        q=mk.get('question') or ''
        if not q.startswith(f"Will the highest temperature in {c} be"): continue
        toks=json.loads(mk.get('clobTokenIds') or '[]'); op=json.loads(mk.get('outcomePrices') or '[]')
        if len(toks)!=2: continue
        out.append(dict(city=c,market_id=str(mk['id']),condition_id=mk.get('conditionId'),question=q,slug=mk.get('slug'),outcome_prices=op,tokens=toks,closed=bool(mk.get('closed')),end_date=(mk.get('endDate') or e.get('endDate')),event_slug=e.get('slug')))
    return out
def prices(m):
    p=f"data/global/prices/{m['market_id']}.json"
    if os.path.exists(p): return 0
    end=int(pd.Timestamp(m['end_date']).timestamp())+86400; start=end-4*86400; hist=[]
    for a in range(start,end,2*86400):
        j=get("https://clob.polymarket.com/prices-history",{"market":m['tokens'][0],"startTs":a,"endTs":min(a+2*86400,end),"fidelity":10})
        if j: hist+=j.get('history',[])
        time.sleep(0.05)
    json.dump({'market_id':m['market_id'],'yes':[{'t':x['t'],'p':x['p']} for x in hist]},open(p,'w')); return len(hist)
days=pd.date_range(FROM,TO); jobs=[(c,d) for c in CITIES for d in days]; print(len(jobs),"city-days",flush=True)
new=[]; n=0
with ThreadPoolExecutor(4) as ex:
    for out in ex.map(lambda cd: event(*cd), jobs):
        for m in out:
            if m['market_id'] not in have: ms.append(m); have.add(m['market_id']); new.append(m)
        n+=1
        if n%200==0: json.dump(ms,open(mpath,'w')); print("events",n,"markets",len(ms),flush=True)
json.dump(ms,open(mpath,'w')); print("markets total",len(ms),flush=True)
todo=[m for m in ms if not os.path.exists(f"data/global/prices/{m['market_id']}.json")]; print("price histories to fetch:",len(todo),flush=True); n=0
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(prices,todo):
        n+=1
        if n%500==0: print("prices",n,flush=True)
print("done",flush=True)
