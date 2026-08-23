import numpy as np, time
from core import *
from scipy_free import fit_scale, apply_scale
import models

S=load_seasons()
BURNIN=45  # days into the season before a prediction counts

def evaluate(signal_fn, name, verbose=False):
    """signal_fn(season) -> per-game signal array. LOSO-calibrated."""
    sig={}; 
    for y,s in S.items():
        sig[y]=np.asarray(signal_fn(s),dtype=float)
    out={}
    P={}
    for y,s in S.items():
        tr_s=np.concatenate([sig[z][ (S[z].day>=BURNIN) ] for z in S if z!=y])
        tr_y=np.concatenate([S[z].y[(S[z].day>=BURNIN)] for z in S if z!=y])
        w=fit_scale(tr_s,tr_y)
        P[y]=apply_scale(sig[y],w)
    res={}
    for tag,maskfn in [('late',lambda s:s.day>=BURNIN),('tourney',lambda s:s.tourney)]:
        p=np.concatenate([P[y][maskfn(S[y])] for y in S])
        yy=np.concatenate([S[y].y[maskfn(S[y])] for y in S])
        res[tag]=report(name,p,yy)
    return res,P,sig

def show(rows,tag):
    print(f"\n== {tag} ==")
    print(fmt(rows))

if __name__=='__main__':
    tests=[]
    t0=time.time()
    # current repo config
    tests.append(('elo repo (K32/20,hca100,538mov)',
        lambda s: models.elo_win(s,K=20,K_early=32,hca=100,use_mov=True,mov_form='538')[0]))
    tests.append(('elo K20 no-mov hca100',
        lambda s: models.elo_win(s,K=20,hca=100,use_mov=False)[0]))
    tests.append(('elo-margin K.10 hca3',
        lambda s: models.elo_margin(s,K=0.10,hca=3.0)[0]))
    tests.append(('ridge margin lam40 cap22',
        lambda s: models.ridge_margin(s,lam=40,cap=22,refit_every=8)[0]))
    late=[];tour=[]
    for name,fn in tests:
        r,_,_=evaluate(fn,name)
        late.append(r['late']); tour.append(r['tourney'])
        print(f"done {name} {time.time()-t0:.1f}s")
    show(late,'games after day 45 (all seasons)')
    show(tour,'NCAA tournament games only')
