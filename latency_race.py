"""Poll Synoptic (the WRH page's backend) and aviationweather every 5 s; log the
wall-clock moment each new observation first appears in each feed."""
import requests, time, json, datetime as dt, sys
TOK='7c76618b66c74aee913bdbae4b448bdd'
H={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36","Referer":"https://www.weather.gov/wrh/timeseries?site=klga","Origin":"https://www.weather.gov","Accept":"application/json"}
ST="KLGA,KMIA,KORD,KDAL,KSEA,KATL,KHOU,KLAX,EGLC,LFPB,EDDM,LEMD,LIMC,EHAM"
seen={}; out=open("out/latency_race.jsonl","a")
end=time.time()+float(sys.argv[1])*60 if len(sys.argv)>1 else time.time()+20*60
s=requests.Session()
while time.time()<end:
    now=dt.datetime.now(dt.timezone.utc)
    try:
        r=s.get("https://api.synopticdata.com/v2/stations/timeseries",params={"STID":ST,"recent":30,"token":TOK,"obtimezone":"utc"},headers=H,timeout=10).json()
        if not r.get('STATION'): print("syn:", r['SUMMARY']['RESPONSE_MESSAGE'], flush=True)
        for st in r.get('STATION',[]):
            o=st['OBSERVATIONS']
            for t,v in zip(o['date_time'],o['air_temp_set_1']):
                k=('synoptic',st['STID'],t)
                if k not in seen and v is not None:
                    seen[k]=now; rec=dict(feed='synoptic',station=st['STID'],obs_time=t,temp=v,first_seen=now.isoformat()); out.write(json.dumps(rec)+"\n"); out.flush(); print(rec,flush=True)
    except Exception as e: print("syn err",e,flush=True)
    try:
        a=s.get("https://aviationweather.gov/api/data/metar",params={"ids":ST,"format":"json","hours":1},timeout=10).json()
        for x in a:
            k=('avwx',x['icaoId'],x['obsTime'])
            if k not in seen:
                seen[k]=now; rec=dict(feed='avwx',station=x['icaoId'],obs_time=dt.datetime.fromtimestamp(x['obsTime'],dt.timezone.utc).isoformat(),temp=x.get('temp'),receipt=x.get('receiptTime'),first_seen=now.isoformat()); out.write(json.dumps(rec)+"\n"); out.flush(); print(rec,flush=True)
    except Exception as e: print("avwx err",e,flush=True)
    time.sleep(5)
