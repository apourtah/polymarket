import requests, time, datetime as dt, json, sys
ST=["EGLC","LFPB","EDDM","LEMD","LIMC","EHAM","EPWA","UUWW","LTAC","KLGA","KORD","KMIA"]
seen={}; s=requests.Session(); end=time.time()+float(sys.argv[1])*60; out=open("out/race_tgftp.jsonl","a")
while time.time()<end:
    now=dt.datetime.now(dt.timezone.utc)
    for st in ST:
        try:
            r=s.get(f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{st}.TXT",timeout=8); l=r.text.strip().splitlines()
            if len(l)>=2:
                k=("tgftp",st,l[0])
                if k not in seen: seen[k]=now; rec=dict(feed="tgftp",station=st,obs=l[0],first_seen=now.isoformat()); out.write(json.dumps(rec)+"\n"); out.flush(); print(rec,flush=True)
        except Exception as e: pass
    try:
        a=s.get("https://aviationweather.gov/api/data/metar",params={"ids":",".join(ST),"format":"json","hours":1},timeout=8).json()
        for x in a:
            k=("avwx",x["icaoId"],x["obsTime"])
            if k not in seen: seen[k]=now; rec=dict(feed="avwx",station=x["icaoId"],obs=dt.datetime.fromtimestamp(x["obsTime"],dt.timezone.utc).strftime("%Y/%m/%d %H:%M"),first_seen=now.isoformat(),receipt=x.get("receiptTime")); out.write(json.dumps(rec)+"\n"); out.flush(); print(rec,flush=True)
    except Exception as e: pass
    time.sleep(5)
