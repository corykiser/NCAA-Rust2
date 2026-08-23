"""Chain seasons so each starts from last season's ratings."""
import numpy as np, pickle
from core import *
from scipy_free import fit_scale, apply_scale
import models
ALL=[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026]
EVAL=YEARS
raw=pickle.load(open('games.pkl','rb'))
SEA={y:Season(y,raw[y]) for y in ALL}
BURNIN=45

def chain_ridge_eff(lam=1.0, prior_w=0.0, regress=0.6, refit_every=24):
    off=None;dfe=None;sig={}
    for y in ALL:
        s=SEA[y]
        sg,off,dfe,h=models.ridge_eff_prior(s,lam=lam,refit_every=refit_every,
                     prior_off=off,prior_def=dfe,prior_w=prior_w,regress=regress)
        sig[y]=sg
    return sig

def chain_elo_margin(K=0.14,hca=3.0,cap=99,regress=0.6):
    prev=None;sig={}
    for y in ALL:
        s=SEA[y]
        init=None if prev is None else {n:regress*v for n,v in prev.items()}
        sg,prev=models.elo_margin(s,K=K,hca=hca,cap=cap,init=init,shrink_early=False)
        sig[y]=sg
    return sig

def score(sig,name):
    P={}
    for y in EVAL:
        tr_s=np.concatenate([sig[z][SEA[z].day>=BURNIN] for z in EVAL if z!=y])
        tr_y=np.concatenate([SEA[z].y[SEA[z].day>=BURNIN] for z in EVAL if z!=y])
        w=fit_scale(tr_s,tr_y); P[y]=apply_scale(sig[y],w)
    out={}
    for tag,mf in [('late',lambda s:s.day>=BURNIN),('early',lambda s:s.day<BURNIN),('tourney',lambda s:s.tourney)]:
        p=np.concatenate([P[y][mf(SEA[y])] for y in EVAL]); yy=np.concatenate([SEA[y].y[mf(SEA[y])] for y in EVAL])
        out[tag]=report(name,p,yy)
    return out
