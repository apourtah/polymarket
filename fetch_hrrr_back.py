"""HRRR 2-m temperature at the 7 US stations from the 00Z and 12Z runs (AWS noaa-hrrr-bdp-pds),
forecast hours 16Z-00Z (covers 12:00-17:00 local from PDT to EDT). -> data/hrrr_t2m.parquet"""
import json, os, sys, datetime as dt, numpy as np, pandas as pd, logging
from concurrent.futures import ThreadPoolExecutor
from herbie import Herbie
logging.getLogger().setLevel(logging.ERROR)
coords=json.load(open('out/station_coords.json')); site=json.load(open('out/stations.json'))
cities=['Houston','Los Angeles','Atlanta','Miami','Austin','Seattle','San Francisco']; st={c:site[c][0] for c in cities}
OUT='data/hrrr_t2m.parquet'; done=set()
if os.path.exists(OUT):
    prev=pd.read_parquet(OUT); done={(r.date,r.run,r.fxx) for r in prev.itertuples()}
else: prev=pd.DataFrame()
idx=None
def grid_index(ds):
    lat=ds.latitude.values; lon=ds.longitude.values-360; out={}
    for c in cities:
        la,lo=coords[st[c]]; d=(lat-la)**2+(lon-lo)**2; out[c]=np.unravel_index(d.argmin(),d.shape)
    return out
def one(args):
    global idx
    date,run,fxx=args
    if (date,run,fxx) in done: return []
    try:
        H=Herbie(f"{date} {run:02d}:00",model='hrrr',product='sfc',fxx=fxx,verbose=False)
        ds=H.xarray("TMP:2 m above ground",remove_grib=True)
        if idx is None: idx=grid_index(ds)
        t=ds.t2m.values; vt=pd.Timestamp(ds.valid_time.values)
        return [dict(date=date,run=run,fxx=fxx,valid_utc=vt,city=c,t2m_f=float(t[idx[c]])*9/5-459.67) for c in cities]
    except Exception as e:
        return [dict(date=date,run=run,fxx=fxx,valid_utc=None,city='ERR',t2m_f=np.nan)]
jobs=[]
d=dt.date(*map(int,os.environ.get('HRRR_START','2026-05-25').split('-')))
while d<=dt.date(*map(int,os.environ.get('HRRR_END','2026-09-17').split('-'))):
    for run,fh in ([(0,range(16,25))] if os.environ.get('HRRR_00Z_ONLY') else [(0,range(16,25)),(12,range(4,13))]):      # 00Z: f16-f24 = 16Z-00Z; 12Z: f04-f12 = 16Z-00Z
        for f in fh: jobs.append((str(d),run,f))
    d+=dt.timedelta(days=1)
print(len(jobs),"jobs;", len(done),"done",flush=True)
rows=[]; n=0
with ThreadPoolExecutor(3) as ex:
    for res in ex.map(one,jobs):
        rows+=res; n+=1
        if n%100==0:
            pd.concat([prev,pd.DataFrame(rows)]).to_parquet(OUT); print(n,flush=True)
pd.concat([prev,pd.DataFrame(rows)]).to_parquet(OUT); print("done",flush=True)
