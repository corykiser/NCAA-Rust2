import numpy as np, pickle, csv
from core import *
from scipy_free import fit_scale, apply_scale
import models, carry
S=load_seasons(); BURNIN=45
X,N=pickle.load(open('feats.pkl','rb'))

def calib(sig,name):
    P={}
    for y in S:
        tr=np.concatenate([sig[z][S[z].day>=BURNIN] for z in S if z!=y])
        ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in S if z!=y])
        w=fit_scale(tr,ty); P[y]=apply_scale(sig[y],w)
    return P

SIG={}
SIG['A. Elo, as shipped in this repo']={y:models.elo_win(S[y],K=20,K_early=32,hca=100,use_mov=True)[0] for y in S}
SIG['B. Elo, wins only (no MOV)']={y:models.elo_win(S[y],K=20,hca=100,use_mov=False)[0] for y in S}
SIG['C. Elo on margin, tuned']={y:models.elo_margin(S[y],K=0.14,hca=3.0,cap=99,shrink_early=False)[0] for y in S}
c=carry.chain_elo_margin(K=0.14,regress=0.9); SIG['D. C + preseason carryover']={y:c[y] for y in S}
SIG['E. Ridge margin (Massey)']={y:models.ridge_margin(S[y],lam=1.0,cap=22,refit_every=8)[0] for y in S}
print('massey done',flush=True)
SIG['F. Ridge adjusted efficiency']={y:X[y][:,N.index('eff_margin')] for y in S}
cp=carry.chain_ridge_eff(lam=1.0,prior_w=5,regress=0.7,refit_every=40)
SIG['G. F + preseason carryover']={y:cp[y] for y in S}
print('prior done',flush=True)
P={k:calib(v,k) for k,v in SIG.items()}

# LightGBM
import lightgbm as lgb
Pg={}
for y in S:
    trm=[z for z in S if z!=y]
    tr=np.concatenate([X[z][S[z].day>=BURNIN] for z in trm]); ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in trm])
    tw=np.concatenate([np.where(S[z].tourney[S[z].day>=BURNIN],6.0,1.0) for z in trm])
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=15,min_child_samples=100,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,verbose=-1).fit(tr,ty,sample_weight=tw)
    Pg[y]=m.predict_proba(X[y])[:,1]
P['H. LightGBM on 19 features']=Pg
print('lgbm done',flush=True)

def trank(y):
    d={}
    for r in csv.reader(open(f'trank/{y}.csv')):
        if len(r)<25: continue
        try: d[r[0].strip()]=(float(r[1]),float(r[2]),float(r[3]),float(r[24]))
        except Exception: pass
    return d
tsig={}
for y,s in S.items():
    T=trank(y); ss=np.zeros(s.n)
    for i in np.where(s.tourney)[0]:
        oa,da,_,ta=T[s.rows[i]['team']]; ob,db,_,tb=T[s.rows[i]['opp']]
        ss[i]=((oa-da)-(ob-db))*(ta+tb)/200.0
    tsig[y]=ss
# calibrate T-Rank on tourney games only (LOSO) since it is a pre-tourney snapshot
Pt={}
for y in S:
    tr=np.concatenate([tsig[z][S[z].tourney] for z in S if z!=y]); ty=np.concatenate([S[z].y[S[z].tourney] for z in S if z!=y])
    w=fit_scale(tr,ty); Pt[y]=apply_scale(tsig[y],w)
P["I. Torvik's published T-Rank"]=Pt

pickle.dump((P,),open('preds.pkl','wb'))
rows={'late':[],'early':[],'tourney':[]}
for k in P:
    for tag,mf in [('late',lambda s:s.day>=BURNIN),('early',lambda s:s.day<BURNIN),('tourney',lambda s:s.tourney)]:
        p=np.concatenate([P[k][y][mf(S[y])] for y in S]); yy=np.concatenate([S[y].y[mf(S[y])] for y in S])
        rows[tag].append(report(k,p,yy))
for tag in ['tourney','late','early']:
    print();print('==',tag,'==');print(fmt(rows[tag]))

# bootstrap vs G on tourney
rng=np.random.default_rng(1)
yy=np.concatenate([S[y].y[S[y].tourney] for y in S]); n=len(yy)
base=np.concatenate([P['F. Ridge adjusted efficiency'][y][S[y].tourney] for y in S])
print('\nΔ tournament logloss vs F (bootstrap 95% CI, 2000 resamples)')
for k in P:
    p=np.concatenate([P[k][y][S[y].tourney] for y in S])
    d=np.array([logloss(p[i],yy[i])-logloss(base[i],yy[i]) for i in (rng.integers(0,n,n) for _ in range(2000))])
    print(f'{k:<34}{d.mean():+.4f}  [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]')
