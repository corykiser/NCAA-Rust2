import numpy as np, math
from core import *
from scipy_free import fit_scale, apply_scale
import models
S=load_seasons(); BURNIN=45

def ridge_eff_w(s, lam=1.0, refit_every=24, halflife=None, full_days=40, decay=0.01, floor=0.6, cap_ppp=None):
    nt=s.nt; poss=s.tempo.copy(); P=2*nt+1
    XtX=np.zeros((P,P)); Xty=np.zeros(P); lam_v=np.full(P,lam); lam_v[-1]=1e-2
    beta=np.zeros(P); since=0; mean=1.02
    sig=np.empty(s.n)
    # weights are applied at *observation* time relative to end of season (a
    # constant per game, so no leakage)
    end=s.day[-1]
    for i in range(s.n):
        if since>=refit_every or i==0:
            beta=models._solve(XtX+np.diag(lam_v),Xty); since=0
        a,b=s.a[i],s.b[i]; h=0.0 if s.neutral[i] else 1.0
        pa=mean+beta[a]+beta[nt+b]+h*beta[-1]; pb=mean+beta[b]+beta[nt+a]-h*beta[-1]
        sig[i]=(pa-pb)*poss[i]
        ya=s.sa[i]/poss[i]-mean; yb=s.sb[i]/poss[i]-mean
        if cap_ppp:
            d=(ya-yb)/2; c=float(np.clip(d,-cap_ppp,cap_ppp)); mid=(ya+yb)/2; ya,yb=mid+c,mid-c
        age=end-s.day[i]
        if halflife: w=0.5**(age/halflife)
        else: w=max(floor,1.0-decay*max(0,age-full_days))
        for (o,dfn,yv,hh) in ((a,b,ya,h),(b,a,yb,-h)):
            j=[o,nt+dfn,P-1]; c=[1.0,1.0,hh]
            for u in range(3):
                for v in range(3): XtX[j[u],j[v]]+=w*c[u]*c[v]
                Xty[j[u]]+=w*c[u]*yv
        since+=1
    return sig

def ev(name,fn):
    sig={y:fn(S[y]) for y in S}
    P={}
    for y in S:
        tr=np.concatenate([sig[z][S[z].day>=BURNIN] for z in S if z!=y]);ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in S if z!=y])
        w=fit_scale(tr,ty);P[y]=apply_scale(sig[y],w)
    o={}
    for tag,mf in [('late',lambda s:s.day>=BURNIN),('tourney',lambda s:s.tourney)]:
        o[tag]=report(name,np.concatenate([P[y][mf(S[y])] for y in S]),np.concatenate([S[y].y[mf(S[y])] for y in S]))
    return o
rows={'late':[],'tourney':[]}
cfgs=[('no weighting',dict(decay=0.0,floor=1.0)),
      ('torvik 40d/1%/0.6',dict()),
      ('halflife 60d',dict(halflife=60)),
      ('halflife 100d',dict(halflife=100)),
      ('torvik + cap ppp .30',dict(cap_ppp=0.30)),
      ('cap ppp .30 only',dict(decay=0.0,floor=1.0,cap_ppp=0.30)),
      ('cap ppp .22 only',dict(decay=0.0,floor=1.0,cap_ppp=0.22))]
for n,kw in cfgs:
    r=ev(n,lambda s,kw=kw: ridge_eff_w(s,**kw))
    for k in rows: rows[k].append(r[k])
    print(n,'done',flush=True)
for k in rows: print();print('==',k,'==');print(fmt(rows[k]))
