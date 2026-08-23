import csv, datetime, os, json
import numpy as np

COL = dict(date=0,team=2,conf=3,opp=4,site=5,result=6,
           adjo=7,adjd=8,ortg=9,efg=10,tov=11,orb=12,ftr=13,
           doppo=14,defg=15,dtov=16,dorb=17,dftr=18,oppconf=20,tempo=23,gid=24,admargin=27)

def load_season(year, path=None):
    path = path or f"tv/{year}.csv"
    rows=[]
    for rec in csv.reader(open(path)):
        if len(rec)<25: continue
        try:
            d=datetime.datetime.strptime(rec[0].strip(),"%m/%d/%y").date()
        except ValueError: continue
        site=rec[5].strip()
        if site not in ("H","A","N"): continue
        res=rec[6].strip()
        try:
            wl,sc=res.split(",");ws,ls=sc.strip().split("-");ws=int(ws);ls=int(ls)
        except Exception: continue
        wl=wl.strip()
        if wl=="W": ts,os_=ws,ls
        elif wl=="L": ts,os_=ls,ws
        else: continue
        def f(i):
            try: return float(rec[i])
            except Exception: return np.nan
        rows.append(dict(date=d,team=rec[2].strip(),conf=rec[3].strip(),opp=rec[4].strip(),
                         oppconf=rec[20].strip(),site=site,ts=ts,os=os_,
                         tempo=f(23),efg=f(10),tov=f(11),orb=f(12),ftr=f(13),
                         defg=f(15),dtov=f(16),dorb=f(17),dftr=f(18),
                         ortg=f(9),drtg=f(14),gid=rec[24].strip(),year=year))
    # reconcile to one row per game, oriented (home=team) or (neutral: first alphabetically)
    games={}
    for r in rows:
        if r['site']=='A': continue
        if r['site']=='N':
            key=(r['date'],*sorted([r['team'],r['opp']]))
            if key in games: continue
            games[key]=r
        else:
            key=(r['date'],r['team'],r['opp'],'H')
            games[key]=r
    out=sorted(games.values(), key=lambda r:(r['date'],r['team'],r['opp']))
    return out

if __name__=="__main__":
    for y in [2017,2018,2019,2021,2022,2023,2024,2025,2026]:
        g=load_season(y)
        n_neu=sum(1 for r in g if r['site']=='N')
        print(y, len(g), 'neutral', n_neu, 'first', g[0]['date'], 'last', g[-1]['date'])
