"""Day-before forecasts for the 37 non-US cities from Open-Meteo previous-runs (previous_day1 = the run issued the day
before): ECMWF IFS 0.25, GFS, ICON temperature (afternoon 11-18 local max) + ECMWF humidity/cloud/wind/pressure fields.
-> data/global/forecasts.parquet ; env FROM (2026-05-25) TO (2026-09-19)"""
import json, re, time, requests, pandas as pd, numpy as np, os
from sweep_local import TZ
FROM=os.environ.get('FROM','2026-05-25'); TO=os.environ.get('TO','2026-09-19')
site=json.load(open('out/stations.json')); coords=json.load(open('out/station_coords.json')); US={'Chicago','Denver','New York City','Dallas','Atlanta','Miami','Austin','Houston','Seattle','Los Angeles','San Francisco'}
CITIES=[c for c in site if c not in US]; S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
MODELS={'ecmwf':'ecmwf_ifs025','gfs':'gfs_seamless','icon':'icon_seamless'}
rows=[]
for c in CITIES:
    lat,lon=coords[site[c][0]]; tz=TZ[c]; parts={}
    for k,m in MODELS.items():
        v=["temperature_2m_previous_day1"]+(["dew_point_2m_previous_day1","relative_humidity_2m_previous_day1","cloud_cover_previous_day1","wind_speed_10m_previous_day1","surface_pressure_previous_day1","shortwave_radiation_previous_day1","precipitation_previous_day1"] if k=='ecmwf' else [])
        for att in range(4):
            r=S.get("https://previous-runs-api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"hourly":",".join(v),"start_date":FROM,"end_date":TO,"models":m,"timezone":tz},timeout=90)
            if r.status_code==200: break
            time.sleep(3*(att+1))
        j=r.json()
        if 'hourly' not in j: print("no data",c,m,str(j)[:100]); continue
        h=pd.DataFrame(j['hourly']); h['time']=pd.to_datetime(h.time); parts[k]=h; time.sleep(0.4)
    if 'ecmwf' not in parts: continue
    h=parts['ecmwf'].copy(); h['day']=h.time.dt.date; h['lh']=h.time.dt.hour; aft=h[h.lh.between(11,18)]; morn=h[h.lh.between(4,8)]
    g=aft.groupby('day').agg(ecmwf=('temperature_2m_previous_day1','max'),dew=('dew_point_2m_previous_day1','mean'),rhum=('relative_humidity_2m_previous_day1','mean'),cloud=('cloud_cover_previous_day1','mean'),wind=('wind_speed_10m_previous_day1','mean'),pres=('surface_pressure_previous_day1','mean'),rad=('shortwave_radiation_previous_day1','mean'),precip=('precipitation_previous_day1','sum'))
    g['t_morn']=morn.groupby('day').temperature_2m_previous_day1.mean(); g['cloud_morn']=morn.groupby('day').cloud_cover_previous_day1.mean()
    for k in ('gfs','icon'):
        if k in parts:
            hh=parts[k]; hh['day']=hh.time.dt.date; hh['lh']=hh.time.dt.hour; g[k]=hh[hh.lh.between(11,18)].groupby('day').temperature_2m_previous_day1.max()
    g['city']=c; rows.append(g.reset_index()); print(c, len(g), "days; ECMWF max sample", g.ecmwf.round(1).tail(3).tolist(), flush=True)
F=pd.concat(rows); F.to_parquet('data/global/forecasts.parquet'); print("saved", len(F), "city-days")
