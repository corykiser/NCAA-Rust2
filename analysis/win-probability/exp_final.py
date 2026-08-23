import numpy as np, pickle, csv, json, datetime
from core import *
from scipy_free import fit_scale, apply_scale
S=load_seasons(); BURNIN=45
X,N=pickle.load(open('feats.pkl','rb'))
rng=np.random.default_rng(0)

# --- seed diff per tourney game (aligned to tourney indices)
def seedmap(y):
    d=json.load(open(f'br/{y}.json'))['championships'][0]['games']; m={}
    for g in d:
        ts=g.get('teams') or []
        if len(ts)!=2: continue
        try: sc=sorted(int(t['score']) for t in ts)
        except Exception: continue
        if sc==[0,0] or not g.get('startDate'): continue
        dt=datetime.datetime.strptime(g['startDate'],"%m/%d/%Y").date()
        m[(dt,sc[0],sc[1])]={int(t['score']):int(t['seed']) for t in ts if t.get('seed')}
    return m
seed_diff={}
for y,s in S.items():
    m=seedmap(y); v=[]
    for i in np.where(s.tourney)[0]:
        key=(s.date[i],*sorted([int(s.sa[i]),int(s.sb[i])])); e=None
        for dd in (0,-1,1): e=m.get((key[0]+datetime.timedelta(days=dd),key[1],key[2])) or e
        if e and int(s.sa[i]) in e and int(s.sb[i]) in e: v.append(float(e[int(s.sb[i])]-e[int(s.sa[i])]))
        else: v.append(0.0)
    seed_diff[y]=np.array(v)

def trank(y):
    d={}
    for r in csv.reader(open(f'trank/{y}.csv')):
        if len(r)<25: continue
        try: d[r[0].strip()]=(float(r[1]),float(r[2]),float(r[3]),float(r[24]))
        except Exception: pass
    return d
tsig={}
for y,s in S.items():
    T=trank(y); ss=[]
    for i in np.where(s.tourney)[0]:
        oa,da,_,ta=T[s.rows[i]['team']]; ob,db,_,tb=T[s.rows[i]['opp']]
        ss.append(((oa-da)-(ob-db))*(ta+tb)/200.0)
    tsig[y]=np.array(ss)

Yt={y:S[y].y[S[y].tourney] for y in S}
def loso_multi(name, feat):
    """feat[y] -> (n_t, k) matrix of tournament-game features; LOSO logistic."""
    from sklearn.linear_model import LogisticRegression
    P=[];Y=[]
    for y in S:
        tr=np.concatenate([feat[z] for z in S if z!=y]); ty=np.concatenate([Yt[z] for z in S if z!=y])
        mu,sd=tr.mean(0),tr.std(0)+1e-9
        lr=LogisticRegression(C=1.0,max_iter=2000).fit((tr-mu)/sd,ty)
        P.append(lr.predict_proba((feat[y]-mu)/sd)[:,1]); Y.append(Yt[y])
    return name,np.concatenate(P),np.concatenate(Y)

col=lambda k:{y:X[y][:,N.index(k)][S[y].tourney] for y in S}
eff=col('eff_margin'); elo=col('elo_margin')
sets=[
 ('constant 0.5',{y:np.zeros((len(Yt[y]),1)) for y in S}),
 ('seed diff',{y:seed_diff[y][:,None] for y in S}),
 ('elo-margin (tuned)',{y:elo[y][:,None] for y in S}),
 ('ridge adjEM (mine)',{y:eff[y][:,None] for y in S}),
 ('Torvik T-Rank adjEM',{y:tsig[y][:,None] for y in S}),
 ('adjEM + seed',{y:np.column_stack([eff[y],seed_diff[y]]) for y in S}),
 ('T-Rank + seed',{y:np.column_stack([tsig[y],seed_diff[y]]) for y in S}),
 ('adjEM + T-Rank + seed',{y:np.column_stack([eff[y],tsig[y],seed_diff[y]]) for y in S}),
 ('adjEM + elo + seed',{y:np.column_stack([eff[y],elo[y],seed_diff[y]]) for y in S}),
]
res=[]
for nm,f in sets:
    nm,p,yy=loso_multi(nm,f); res.append((nm,p,yy))
print(fmt([report(nm,p,yy) for nm,p,yy in res]))
# bootstrap CI of the difference vs 'ridge adjEM (mine)'
base=dict((nm,p) for nm,p,_ in res)['ridge adjEM (mine)']
yy=res[0][2]; n=len(yy)
print('\nbootstrap 95% CI on Δlogloss vs ridge adjEM (2000 resamples of the 600 games)')
for nm,p,_ in res:
    d=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n)
        d.append(logloss(p[idx],yy[idx])-logloss(base[idx],yy[idx]))
    d=np.array(d); print(f'{nm:<28}{np.mean(d):+.4f}  [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]')
