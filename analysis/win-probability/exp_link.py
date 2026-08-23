import numpy as np, pickle, csv, math, json, datetime
from core import *
from scipy_free import fit_scale, apply_scale
S=load_seasons(); BURNIN=45
X,N=pickle.load(open('feats.pkl','rb'))
J=N.index('eff_margin')
sig={y:X[y][:,J] for y in S}

def loso(sig,name,link='logit',mult_tourney=1.0,clip=None):
    P={}
    for y in S:
        tr=np.concatenate([sig[z][S[z].day>=BURNIN] for z in S if z!=y])
        ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in S if z!=y])
        sg=sig[y].copy()
        sg=np.where(S[y].tourney,sg*mult_tourney,sg)
        if link=='logit':
            w=fit_scale(tr,ty); p=apply_scale(sg,w)
        else:  # probit, fit sigma by grid
            from math import erf
            def phi(z): return 0.5*(1+np.vectorize(math.erf)(z/np.sqrt(2)))
            best=(1e9,None)
            for s_ in np.arange(8.0,16.01,0.25):
                ll=logloss(np.clip(phi(tr/s_),1e-9,1-1e-9),ty)
                if ll<best[0]: best=(ll,s_)
            p=phi(sg/best[1])
        if clip: p=np.clip(p,clip,1-clip)
        P[y]=p
    out={}
    for tag,mf in [('late',lambda s:s.day>=BURNIN),('tourney',lambda s:s.tourney)]:
        pp=np.concatenate([P[y][mf(S[y])] for y in S]); yy=np.concatenate([S[y].y[mf(S[y])] for y in S])
        out[tag]=report(name,pp,yy)
    return out,P

rows={'late':[],'tourney':[]}
for nm,kw in [('eff logit',dict()),('eff probit',dict(link='probit')),
              ('eff logit x1.07 tourney',dict(mult_tourney=1.07)),
              ('eff logit x1.15 tourney',dict(mult_tourney=1.15)),
              ('eff logit x0.93 tourney',dict(mult_tourney=0.93)),
              ('eff logit clip .02',dict(clip=0.02)),
              ('eff logit clip .05',dict(clip=0.05))]:
    r,_=loso(sig,nm,**kw)
    for k in rows: rows[k].append(r[k])
for k in rows: print();print('==',k,'==');print(fmt(rows[k]))

# ---------------- ensemble with Torvik pre-tourney T-Rank, tournament games only
def trank(y):
    d={}
    for r in csv.reader(open(f'trank/{y}.csv')):
        if len(r)<25: continue
        try: d[r[0].strip()]=(float(r[1]),float(r[2]),float(r[3]),float(r[24]))
        except Exception: pass
    return d
tsig={}; tk={}
for y,s in S.items():
    T=trank(y); ss=[]
    for i in np.where(s.tourney)[0]:
        na,nb=s.rows[i]['team'],s.rows[i]['opp']
        oa,da,_,ta=T[na]; ob,db,_,tb=T[nb]
        ss.append(((oa-da)-(ob-db))*(ta+tb)/200.0)
    tsig[y]=np.array(ss)
def ev(name,S2):
    P=[];Y=[]
    for y in S:
        tr=np.concatenate([S2[z] for z in S if z!=y]); ty=np.concatenate([S[z].y[S[z].tourney] for z in S if z!=y])
        w=fit_scale(tr,ty); P.append(apply_scale(S2[y],w)); Y.append(S[y].y[S[y].tourney])
    return report(name,np.concatenate(P),np.concatenate(Y))
mine={y:sig[y][S[y].tourney] for y in S}
elo={y:X[y][:,N.index('elo_margin')][S[y].tourney] for y in S}
out=[ev('mine: ridge adjEM',mine), ev('Torvik T-Rank adjEM',tsig), ev('elo-margin',elo)]
for a in [0.25,0.5,0.75]:
    out.append(ev(f'blend {a:.2f}*mine + {1-a:.2f}*Torvik',{y:a*mine[y]/np.std(np.concatenate(list(mine.values())))+(1-a)*tsig[y]/np.std(np.concatenate(list(tsig.values()))) for y in S}))
out.append(ev('blend 0.5 mine + 0.5 elo',{y:0.5*mine[y]/np.std(np.concatenate(list(mine.values())))+0.5*elo[y]/np.std(np.concatenate(list(elo.values()))) for y in S}))
print();print('== tournament ensemble ==');print(fmt(out))
