"""Regional/high-res day-before models for the 37 cities from Open-Meteo previous-runs (temperature_2m_previous_day1,
afternoon max 11-18 local). Per city a list of candidate models; keep whichever return data. -> data/global/regional.parquet"""
import json, time, requests, pandas as pd, os
from sweep_local import TZ
site=json.load(open('out/stations.json')); coords=json.load(open('out/station_coords.json')); S=requests.Session(); S.headers['User-Agent']='Mozilla/5.0'
EU=['icon_eu','icon_d2','meteofrance_arome_france_hd','meteofrance_arpege_europe','ukmo_uk_deterministic_2km','ukmo_global_deterministic_10km','knmi_harmonie_arome_europe','dmi_harmonie_arome_europe','italia_meteo_arpae_icon_2i','metno_nordic']
REG={'London':EU,'Paris':EU,'Munich':EU,'Milan':EU,'Madrid':EU,'Warsaw':EU,'Amsterdam':EU,'Helsinki':EU,'Moscow':['icon_eu','ukmo_global_deterministic_10km','meteofrance_arpege_europe'],'Istanbul':['icon_eu','meteofrance_arpege_europe','ukmo_global_deterministic_10km'],'Ankara':['icon_eu','meteofrance_arpege_europe','ukmo_global_deterministic_10km'],
     'Tokyo':['jma_msm','jma_gsm','cma_grapes_global','ukmo_global_deterministic_10km'],'Busan':['jma_msm','jma_gsm','kma_ldps','kma_gdps','cma_grapes_global'],'Seoul (Incheon)':['kma_ldps','kma_gdps','jma_msm','jma_gsm','cma_grapes_global'],
     'Shanghai':['cma_grapes_global','jma_gsm','ukmo_global_deterministic_10km'],'Beijing':['cma_grapes_global','jma_gsm','ukmo_global_deterministic_10km'],'Wuhan':['cma_grapes_global','ukmo_global_deterministic_10km'],'Chengdu':['cma_grapes_global','ukmo_global_deterministic_10km'],'Chongqing':['cma_grapes_global','ukmo_global_deterministic_10km'],'Shenzhen':['cma_grapes_global','ukmo_global_deterministic_10km'],'Guangzhou':['cma_grapes_global','ukmo_global_deterministic_10km'],'Qingdao':['cma_grapes_global','jma_gsm','ukmo_global_deterministic_10km'],'Zhengzhou':['cma_grapes_global','ukmo_global_deterministic_10km'],
     'Wellington':['bom_access_global','ukmo_global_deterministic_10km'],'Singapore':['ukmo_global_deterministic_10km','cma_grapes_global','meteofrance_arpege_world'],'Kuala Lumpur':['ukmo_global_deterministic_10km','cma_grapes_global','meteofrance_arpege_world'],'Manila':['jma_gsm','ukmo_global_deterministic_10km','cma_grapes_global'],'Lucknow':['ukmo_global_deterministic_10km','cma_grapes_global','meteofrance_arpege_world'],'Karachi':['ukmo_global_deterministic_10km','meteofrance_arpege_world','cma_grapes_global'],'Jeddah':['ukmo_global_deterministic_10km','meteofrance_arpege_world','cma_grapes_global'],'Tel Aviv':['icon_eu','ukmo_global_deterministic_10km','meteofrance_arpege_europe'],'Cape Town':['ukmo_global_deterministic_10km','meteofrance_arpege_world','cma_grapes_global'],
     'Sao Paulo':['ukmo_global_deterministic_10km','meteofrance_arpege_world','cma_grapes_global'],'Buenos Aires':['ukmo_global_deterministic_10km','meteofrance_arpege_world','cma_grapes_global'],'Toronto':['gem_regional','gem_hrdps_continental','gem_global','ukmo_global_deterministic_10km'],'Mexico City':['gem_global','ukmo_global_deterministic_10km','meteofrance_arpege_world'],'Panama City':['gem_global','ukmo_global_deterministic_10km','meteofrance_arpege_world']}
rows=[]
for c,models in REG.items():
    lat,lon=coords[site[c][0]]; got=[]
    for m in models:
        try: r=S.get("https://previous-runs-api.open-meteo.com/v1/forecast",params={"latitude":lat,"longitude":lon,"hourly":"temperature_2m_previous_day1","start_date":"2026-05-25","end_date":"2026-09-19","models":m,"timezone":TZ[c]},timeout=90); j=r.json()
        except Exception as e: j={}
        if 'hourly' not in j or all(v is None for v in j['hourly']['temperature_2m_previous_day1'][:48]): time.sleep(0.3); continue
        h=pd.DataFrame(j['hourly']); h['time']=pd.to_datetime(h.time); h['day']=h.time.dt.date; h['lh']=h.time.dt.hour
        g=h[h.lh.between(11,18)].groupby('day').temperature_2m_previous_day1.max().rename(m); got.append(g); time.sleep(0.3)
    if got:
        G=pd.concat(got,axis=1).reset_index(); G['city']=c; rows.append(G); print(c, [x.name for x in got], flush=True)
    else: print(c,"no regional data",flush=True)
R=pd.concat(rows); R.to_parquet('data/global/regional.parquet'); print("saved", R.shape)
