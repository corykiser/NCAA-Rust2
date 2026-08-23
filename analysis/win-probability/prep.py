import json, datetime, pickle, collections
from load import load_season
YEARS=[2017,2018,2019,2021,2022,2023,2024,2025,2026]

def tourney_keys(year):
    d=json.load(open(f"br/{year}.json"))
    ch=d['championships'][0]
    out=set()
    for g in ch['games']:
        ts=g.get('teams') or []
        if len(ts)!=2: continue
        try: s=sorted(int(t['score']) for t in ts)
        except Exception: continue
        if s[0]==0 and s[1]==0: continue
        sd=g.get('startDate')
        if not sd: continue
        dt=datetime.datetime.strptime(sd,"%m/%d/%Y").date()
        out.add((dt,s[0],s[1]))
    return out

def build():
    all_games={}
    for y in YEARS:
        g=load_season(y)
        tk=tourney_keys(y)
        hits=0
        for r in g:
            key=(r['date'],*sorted([r['ts'],r['os']]))
            # tournament games can be a day off in either source; allow +-1 day
            is_t = any((key[0]+datetime.timedelta(days=dd),key[1],key[2]) in tk for dd in (0,-1,1))
            r['tourney']= is_t and r['site']=='N' and r['date'].month>=3
            hits+=r['tourney']
        print(y,'tourney games matched',hits,'of',len(tk))
        all_games[y]=g
    pickle.dump(all_games,open('games.pkl','wb'))
build()
