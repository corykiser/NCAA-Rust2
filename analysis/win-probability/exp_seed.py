import json, datetime, numpy as np
from core import *
from scipy_free import fit_scale, apply_scale
S=load_seasons()
def seedmap(y):
    d=json.load(open(f'br/{y}.json'))['championships'][0]['games']
    m={}
    for g in d:
        ts=g.get('teams') or []
        if len(ts)!=2: continue
        try: sc=sorted(int(t['score']) for t in ts)
        except Exception: continue
        if sc==[0,0]: continue
        sd=g.get('startDate')
        if not sd: continue
        dt=datetime.datetime.strptime(sd,"%m/%d/%Y").date()
        # key by (date,scores) -> {score: seed}
        m[(dt,sc[0],sc[1])]={int(t['score']):int(t['seed']) for t in ts if t.get('seed')}
    return m
sig={};ys={}
for y,s in S.items():
    m=seedmap(y); ss=[];yy=[]
    for i in np.where(s.tourney)[0]:
        key=(s.date[i],*sorted([int(s.sa[i]),int(s.sb[i])]))
        e=None
        for dd in (0,-1,1):
            e=m.get((key[0]+datetime.timedelta(days=dd),key[1],key[2])) or e
        if not e or len(e)<2: continue
        sa=e.get(int(s.sa[i])); sb=e.get(int(s.sb[i]))
        if sa is None or sb is None: continue
        ss.append(float(sb-sa)); yy.append(s.y[i])
    sig[y]=np.array(ss); ys[y]=np.array(yy)
P=[];Y=[]
for y in S:
    tr=np.concatenate([sig[z] for z in S if z!=y]); ty=np.concatenate([ys[z] for z in S if z!=y])
    w=fit_scale(tr,ty); P.append(apply_scale(sig[y],w)); Y.append(ys[y])
print(fmt([report('seed difference (tourney)',np.concatenate(P),np.concatenate(Y))]))
