"""HRRR 2-m temperature, 00Z run, f01-f31, at 8 US stations (adds NYC/KLGA) -> data/hrrr_t2m_night.parquet.
Covers the full local calendar day of the run date (04Z-07Z start .. 04Z-07Z next day) for min-temperature markets."""
import json, os, sys, datetime as dt, numpy as np, pandas as pd, logging
from concurrent.futures import ThreadPoolExecutor
from herbie import Herbie
logging.getLogger().setLevel(logging.ERROR)
coords=json.load(open('out/station_coords.json')); site=json.load(open('out/stations.json'))
cities=['Houston','Los Angeles','Atlanta','Miami','Austin','Seattle','San Francisco','New York City']; st={c:site[c][0] for c in cities}
OUT='data/hrrr_t2m_night.parquet'; done=set()
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
d0=dt.date(*map(int,os.environ.get('HRRR_START','2026-05-25').split('-'))); d1=dt.date(2026,9,17)
for fh in [list(range(1,16))+list(range(25,32)), list(range(16,25))]:     # night/morning hours first, afternoon (for NYC) second
    d=d0
    while d<=d1:
        for f in fh: jobs.append((str(d),0,f))
        d+=dt.timedelta(days=1)
print(len(jobs),"jobs;", len(done),"done",flush=True)
rows=[]; n=0
with ThreadPoolExecutor(3) as ex:
    for res in ex.map(one,jobs):
        rows+=res; n+=1
        if n%100==0:
            pd.concat([prev,pd.DataFrame(rows)]).to_parquet(OUT); print(n,flush=True)
pd.concat([prev,pd.DataFrame(rows)]).to_parquet(OUT); print("done",flush=True)
