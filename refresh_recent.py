"""Extend the archives the backtest reads (wx previous-runs, IEM MOS, METAR, markets + CLOB price histories) for the last days,
for the 7 production cities, so backtest_evening.py can be run on recent nights.  env: FROM (2026-09-12) TO (today)"""
import pandas as pd, numpy as np, json, os, re, time, datetime as dt, requests
from zoneinfo import ZoneInfo
from sweep_local import TZ
CITIES=['Los Angeles','Austin','Chicago','Houston','Dallas','Seattle','Miami']; site=json.load(open('out/stations.json')); coords=json.load(open('out/station_coords.json'))
FROM=os.environ.get('FROM','2026-09-12'); TO=os.environ.get('TO',dt.date.today().isoformat()); S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
# 1. Open-Meteo previous-runs (previous_day1 = the run from the day before), same fields/units as data/hrrr_wx_d1.parquet
V=['temperature_2m','dew_point_2m','relative_humidity_2m','wind_speed_10m','wind_direction_10m','cloud_cover','shortwave_radiation','surface_pressure','precipitation']
w=pd.read_parquet('data/hrrr_wx_d1.parquet'); new=[]
for c in CITIES:
    lat,lon=coords[site[c][0]]
    j=S.get("https://previous-runs-api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"hourly":",".join(v+"_previous_day1" for v in V),"start_date":FROM,"end_date":TO,"models":"gfs_hrrr","timezone":TZ[c],"temperature_unit":"fahrenheit","wind_speed_unit":"mph"},timeout=60).json()
    h=pd.DataFrame({'time':pd.to_datetime(j['hourly']['time']),**{v:j['hourly'][v+'_previous_day1'] for v in V}}); h['city']=c; new.append(h); time.sleep(0.3)
new=pd.concat(new).dropna(subset=['temperature_2m']); w=pd.concat([w[~(w.city.isin(CITIES)&(w.time>=pd.Timestamp(FROM)))],new]).sort_values(['city','time']); w.to_parquet('data/hrrr_wx_d1.parquet'); print("wx:", new.time.min(), new.time.max(), len(new))
# 2. IEM MOS (NBS, GFS) last days
mo=pd.read_parquet('data/mos_nbm.parquet'); rows=[]
for c in CITIES:
    for model in ['NBS','GFS']:
        r=S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py",params={"station":site[c][0],"model":model,"sts":f"{FROM}T00:00Z","ets":f"{TO}T23:00Z","format":"csv"},timeout=120)
        if r.status_code==200 and len(r.text)>100:
            import io; d=pd.read_csv(io.StringIO(r.text)); d['city']=c; rows.append(d)
        time.sleep(0.3)
d=pd.concat(rows); d['runtime']=pd.to_datetime(d.runtime,utc=True).dt.tz_localize(None).astype(str); d['ftime']=pd.to_datetime(d.ftime,utc=True).dt.tz_localize(None).astype(str)
keep=[c for c in mo.columns if c in d.columns]; d=d[keep]
mo=pd.concat([mo[~(mo.city.isin(CITIES)&(mo.runtime>=FROM))],d]).drop_duplicates(['runtime','ftime','model','station']); mo.to_parquet('data/mos_nbm.parquet'); print("mos:", d.runtime.min(), d.runtime.max(), len(d))
# 3. METAR (IEM ASOS, report_type 3,4), T-group tenths
# IEM answers an over-quota request with an EMPTY 200, not an error.  Taken at face value that yields zero rows,
# which the merge below then writes OVER good history -- this silently truncated the 5 non-refreshed cities to
# 2026-09-13.  So: retry until the response is plausibly complete, and only replace rows for cities we did refetch.
m=pd.read_parquet('data/metar.parquet'); rows=[]; got=[]
for c in CITIES:
    icao,unit=site[c]; stid=icao[1:] if icao.startswith('K') else icao; f=dt.date.fromisoformat(FROM); t=dt.date.fromisoformat(TO)+dt.timedelta(days=1)
    r=""
    for attempt in range(8):
        try:
            r=S.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",params={"station":stid,"data":"metar","year1":f.year,"month1":f.month,"day1":f.day,"year2":t.year,"month2":t.month,"day2":t.day,"tz":TZ[c],"format":"onlycomma","report_type":"3,4"},timeout=600).text
            if r.count("\n")>=20*max(1,(t-f).days): break
            raise IOError(f"short response ({r.count(chr(10))} lines)")
        except Exception as ex:
            print(f"  retry {c}: {ex.__class__.__name__} {ex}",flush=True); r=""; time.sleep(5*(attempt+1))
    if not r:
        print(f"  GAVE UP {c} -- keeping its existing rows",flush=True); continue
    got.append(c)
    for line in r.splitlines():
        if line.startswith("station,") or not line.strip(): continue
        p=line.split(",",2)
        if len(p)<3: continue
        raw=p[2]; body=re.search(r"\s(M?\d\d)/(M?\d\d)?\s",raw); tg=re.search(r"\sT(\d)(\d{3})(\d)(\d{3})",raw)
        if not body: continue
        cc=(-1 if tg.group(1)=="1" else 1)*int(tg.group(2))/10 if tg else float(body.group(1).replace("M","-"))
        rows.append(dict(city=c,station=icao,unit=unit,local_time=pd.Timestamp(p[1]),temp_c=cc,has_tenths=bool(tg)))
    time.sleep(0.3)
d=pd.DataFrame(rows)
if got:
    m=pd.concat([m[~(m.city.isin(got)&(m.local_time>=pd.Timestamp(FROM)))],d]).sort_values(['station','local_time']); m.to_parquet('data/metar.parquet')
    print("metar:", d.local_time.min(), d.local_time.max(), len(d), f"({len(got)}/{len(CITIES)} cities)")
else: print("metar: nothing fetched, archive left untouched")
# 4. markets + price histories (highest temperature, 7 cities, FROM..TO), same schema as data/markets.json / data/prices
# Ask for each event by SLUG.  The old title_search with limit=100 silently dropped whole days once a city had more
# than 100 events indexed -- 2026-09-21/22/24/25 were never fetched at all -- whereas the slug is fully determined
# by city and date, so the range can just be enumerated.  Markets already held are still re-checked for resolution:
# one first seen while it was open would otherwise keep closed=False for ever, which is what made the recent days
# invisible to every backtest that filters on it.
ms=json.load(open('data/markets.json')); have={x['market_id']:x for x in ms}; added=0; resolved=0
slugify=lambda c: c.lower().replace(' ','-')
d0=dt.date.fromisoformat(FROM); d1=dt.date.fromisoformat(TO)
for day in [d0+dt.timedelta(days=i) for i in range((d1-d0).days+1)]:
    for c in CITIES:
        # Slug spellings seen in the archive: bare before 2026-02-03, year-suffixed after, and a handful of days
        # (2026-05-17..19) carry an "arch-" prefix. Try each; the first that resolves wins.
        stem=f"highest-temperature-in-{slugify(c)}-on-{day.strftime('%B').lower()}-{day.day}"
        ev=None
        for slug in (f"{stem}-{day.year}", stem, f"arch-{stem}-{day.year}", f"arch-{stem}"):
            try: ev=S.get("https://gamma-api.polymarket.com/events",params={"slug":slug},timeout=60).json()
            except Exception as ex: print(f"  event fetch failed {slug}: {ex.__class__.__name__}",flush=True); ev=None
            if ev: break
        if not ev: continue
        e=ev[0]
        for mk in e.get('markets',[]):
            q=mk.get('question') or ''
            if not q.startswith(f"Will the highest temperature in {c} be"): continue
            toks=json.loads(mk.get('clobTokenIds') or '[]'); op=json.loads(mk.get('outcomePrices') or '[]')
            if len(toks)!=2: continue
            mid=str(mk['id'])
            if mid in have:                      # already stored: the only thing that can still change is resolution
                old=have[mid]
                if not old.get('closed') and mk.get('closed'):
                    old['closed']=True; old['outcome_prices']=op; old['uma_status']=mk.get('umaResolutionStatus'); old['closed_time']=mk.get('closedTime')
                    if op and float(op[0]) in (0.0,1.0):
                        losing='Yes' if float(op[0])==0.0 else 'No'
                        try:                     # stored series is the LOSING token's; flip it if YES turned out to win
                            h=json.load(open(f"data/prices/{mid}.json"))
                            if h.get('losing_outcome')!=losing:
                                h['losing_outcome']=losing; h['token']=toks[0] if losing=='Yes' else toks[1]
                                h['history']=[{'t':x['t'],'p':round(1-x['p'],4)} for x in h['history']]
                                json.dump(h,open(f"data/prices/{mid}.json",'w'))
                        except FileNotFoundError: pass
                    resolved+=1
                continue
            rec=dict(event_id=e['id'],event_slug=e.get('slug'),event_title=e.get('title'),event_end=e.get('endDate'),market_id=mid,condition_id=mk.get('conditionId'),question=q,slug=mk.get('slug'),outcomes=["Yes","No"],outcome_prices=op,tokens=toks,closed=bool(mk.get('closed')),uma_status=mk.get('umaResolutionStatus'),created_at=mk.get('createdAt'),start_date=mk.get('startDate'),end_date=mk.get('endDate') or e.get('endDate'),closed_time=mk.get('closedTime'),volume=mk.get('volumeNum'),liquidity=mk.get('liquidityNum'),tags=['weather'])
            # price history of the YES token, 1-min, from 2 days before the end date
            ts_end=int(pd.Timestamp(rec['end_date']).timestamp())+86400; ts_start=ts_end-4*86400
            hist=[]
            for a in range(ts_start,ts_end,86400*2):
                try:
                    j=S.get("https://clob.polymarket.com/prices-history",params={"market":toks[0],"startTs":a,"endTs":min(a+86400*2,ts_end),"fidelity":1},timeout=60).json(); hist+=j.get('history',[])
                except Exception as ex: print(f"    price fetch failed {mid}: {ex.__class__.__name__}",flush=True)
                time.sleep(0.15)
            if not hist: continue
            # store as the LOSING token's series (backtest convention); for unresolved markets store YES and mark it open
            if rec['closed'] and op and float(op[0]) in (0.0,1.0):
                losing='Yes' if float(op[0])==0.0 else 'No'; series=[{'t':x['t'],'p':x['p'] if losing=='Yes' else round(1-x['p'],4)} for x in hist]
            else: losing='Yes'; series=[{'t':x['t'],'p':x['p']} for x in hist]; rec['closed']=False
            json.dump({'market_id':mid,'token':toks[0] if losing=='Yes' else toks[1],'losing_outcome':losing,'history':series},open(f"data/prices/{mid}.json",'w'))
            ms.append(rec); have[mid]=rec; added+=1
        time.sleep(0.2)
json.dump(ms,open('data/markets.json','w')); print("markets added:", added, "| newly resolved:", resolved)
