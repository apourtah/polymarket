"""Day-before HRRR: 12Z run f28-f36 and 18Z run f22-f30 (both = 16Z-00Z of the NEXT day, i.e. the target afternoon) at the
7 production stations, AWS noaa-hrrr-bdp-pds via herbie. -> data/hrrr_t2m_daybefore.parquet  (env HRRR_START/HRRR_END)"""
import json, os, datetime as dt, numpy as np, pandas as pd, logging
from concurrent.futures import ThreadPoolExecutor
from herbie import Herbie
logging.getLogger().setLevel(logging.ERROR)
coords=json.load(open('out/station_coords.json')); site=json.load(open('out/stations.json'))
cities=['Los Angeles','Austin','Chicago','Houston','Dallas','Seattle','Miami']; st={c:site[c][0] for c in cities}
OUT='data/hrrr_t2m_daybefore.parquet'; done=set(); prev=pd.DataFrame()
if os.path.exists(OUT): prev=pd.read_parquet(OUT); done={(r.date,r.run,r.fxx) for r in prev.itertuples()}
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
    except Exception:
        return [dict(date=date,run=run,fxx=fxx,valid_utc=None,city='ERR',t2m_f=np.nan)]
jobs=[]
d=dt.date.fromisoformat(os.environ.get('HRRR_START','2026-01-17'))
while d<=dt.date.fromisoformat(os.environ.get('HRRR_END','2026-09-18')):
    for run,fh in [(12,range(28,37)),(18,range(22,31))]:
        for f in fh: jobs.append((str(d),run,f))
    d+=dt.timedelta(days=1)
print(len(jobs),"jobs;",len(done),"done",flush=True)
rows=[]; n=0
with ThreadPoolExecutor(3) as ex:
    for res in ex.map(one,jobs):
        rows+=res; n+=1
        if n%100==0: pd.concat([prev,pd.DataFrame(rows)]).to_parquet(OUT); print(n,flush=True)
pd.concat([prev,pd.DataFrame(rows)]).to_parquet(OUT); print("done",flush=True)
