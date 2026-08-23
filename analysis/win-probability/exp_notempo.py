"""How much of model F depends on the tempo column actually being there?"""
import numpy as np
from core import *
from scipy_free import fit_scale, apply_scale
import models
S=load_seasons(); BURNIN=45

def ev(name,fn):
    sig={y:fn(S[y]) for y in S}; P={}
    for y in S:
        tr=np.concatenate([sig[z][S[z].day>=BURNIN] for z in S if z!=y])
        ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in S if z!=y])
        w=fit_scale(tr,ty); P[y]=apply_scale(sig[y],w)
    o={}
    for tag,mf in [('late',lambda s:s.day>=BURNIN),('tourney',lambda s:s.tourney)]:
        o[tag]=report(name,np.concatenate([P[y][mf(S[y])] for y in S]),
                          np.concatenate([S[y].y[mf(S[y])] for y in S]))
    return o

def const_tempo(s,val):
    import copy
    t=s.tempo.copy(); s.tempo=np.full(s.n,val)
    try: out=models.ridge_eff(s,lam=1.0,refit_every=24)[0]
    finally: s.tempo=t
    return out

rows={'late':[],'tourney':[]}
for n,fn in [
    ('F: real per-game tempo',      lambda s: models.ridge_eff(s,lam=1.0,refit_every=24)[0]),
    ('F: constant tempo 68',        lambda s: const_tempo(s,68.0)),
    ('E: ridge on margin only',     lambda s: models.ridge_margin(s,lam=1.0,cap=22,refit_every=8)[0]),
    ('E: ridge on margin, no cap',  lambda s: models.ridge_margin(s,lam=1.0,cap=999,refit_every=8)[0]),
]:
    r=ev(n,fn)
    for k in rows: rows[k].append(r[k])
    print(n,'done',flush=True)
for k in rows: print();print('==',k,'==');print(fmt(rows[k]))
