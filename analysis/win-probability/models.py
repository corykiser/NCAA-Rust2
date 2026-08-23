"""Walk-forward rating models. Each returns a per-game 'signal' produced from
information strictly prior to that game, plus the fitted team ratings at the
end of the regular season (for tournament prediction)."""
import numpy as np, math

def _solve(A,b):
    try:
        return np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A,b,rcond=None)[0]

# ---------------------------------------------------------------- Elo (wins)
def elo_win(s, K=20.0, hca=100.0, use_mov=True, K_early=None, early_games=10,
            init=None, scale=400.0, mov_form='538'):
    """Classic Elo. Returns (signal=rating diff incl. hca, final ratings dict)."""
    nt=s.nt
    R=np.full(nt,1500.0)
    if init is not None:
        for name,v in init.items():
            if name in s.idx: R[s.idx[name]]=v
    gp=np.zeros(nt,dtype=int)
    sig=np.empty(s.n)
    for i in range(s.n):
        a,b=s.a[i],s.b[i]
        h=0.0 if s.neutral[i] else hca
        d=R[a]-R[b]+h
        sig[i]=d
        e=1.0/(1.0+10.0**(-d/scale))
        y=s.y[i]
        k=K
        if K_early is not None:
            k=0.5*((K_early if gp[a]<early_games else K)+(K_early if gp[b]<early_games else K))
        m=1.0
        if use_mov:
            marg=abs(s.margin[i])
            # winner's rating edge, home-adjusted
            wd = d if y>0 else -d
            if mov_form=='538':
                m=math.log(marg+3.0)/math.log(3.0)
                m*= 2.2/(max(wd,0.0)*0.001+2.2) if wd>0 else 1.0
            elif mov_form=='nba':   # 538 NBA form
                m=((marg+3.0)**0.8)/(7.5+0.006*wd)
            elif mov_form=='sqrt':
                m=math.sqrt(marg)/math.sqrt(11.0)
        delta=k*m*(y-e)
        R[a]+=delta; R[b]-=delta
        gp[a]+=1; gp[b]+=1
    return sig, {n:R[s.idx[n]] for n in s.names}

# ------------------------------------------------------- Elo on point margin
def elo_margin(s, K=0.10, hca=3.0, cap=22.0, init=None, shrink_early=True):
    """Rating measured in points. Update on margin error (a scalar Kalman-ish
    filter on point differential)."""
    nt=s.nt
    R=np.zeros(nt)
    if init is not None:
        for name,v in init.items():
            if name in s.idx: R[s.idx[name]]=v
    gp=np.zeros(nt,dtype=int)
    sig=np.empty(s.n)
    for i in range(s.n):
        a,b=s.a[i],s.b[i]
        h=0.0 if s.neutral[i] else hca
        pred=R[a]-R[b]+h
        sig[i]=pred
        m=float(np.clip(s.margin[i],-cap,cap))
        err=m-pred
        k=K
        if shrink_early:
            k=K*max(1.0, 3.0/ (1.0+0.35*min(gp[a],gp[b])) )
        R[a]+=0.5*k*err; R[b]-=0.5*k*err
        gp[a]+=1; gp[b]+=1
    return sig, {n:R[s.idx[n]] for n in s.names}

# -------------------------------------------------- Ridge least squares (Massey)
def ridge_margin(s, lam=40.0, cap=22.0, refit_every=1, prior=None, prior_w=0.0,
                 recency_halflife=None, mov_transform=None):
    """Solve  margin ~ R[a]-R[b]+hca  by ridge, refit walk-forward.
    Signal for game i is the fit using games 0..i-1 only."""
    nt=s.nt
    P=nt+1   # +1 for hca column
    XtX=np.zeros((P,P)); Xty=np.zeros(P)
    if prior is not None and prior_w>0:
        for name,v in prior.items():
            if name in s.idx:
                j=s.idx[name]; XtX[j,j]+=prior_w; Xty[j]+=prior_w*v
    sig=np.empty(s.n)
    beta=np.zeros(P)
    lam_v=np.full(P,lam); lam_v[-1]=1e-6
    dirty=True; since=0
    for i in range(s.n):
        if dirty and since>=refit_every-1:
            A=XtX+np.diag(lam_v)
            A[:nt,:nt]+=1e-9
            # center: add a constraint that ratings sum to 0 via penalty (ridge already does)
            try: beta=np.linalg.solve(A,Xty)
            except np.linalg.LinAlgError: beta=np.linalg.lstsq(A,Xty,rcond=None)[0]
            dirty=False; since=0
        a,b=s.a[i],s.b[i]
        h=0.0 if s.neutral[i] else 1.0
        sig[i]=beta[a]-beta[b]+h*beta[-1]
        m=float(np.clip(s.margin[i],-cap,cap))
        if mov_transform=='sqrt':
            m=math.copysign(math.sqrt(abs(m))*math.sqrt(11.0),m)
        w=1.0
        if recency_halflife:
            w=0.5**((s.day[-1]-s.day[i])/recency_halflife)
        XtX[a,a]+=w; XtX[b,b]+=w; XtX[a,b]-=w; XtX[b,a]-=w
        XtX[a,-1]+=w*h; XtX[-1,a]+=w*h; XtX[b,-1]-=w*h; XtX[-1,b]-=w*h
        XtX[-1,-1]+=w*h*h
        Xty[a]+=w*m; Xty[b]-=w*m; Xty[-1]+=w*h*m
        dirty=True; since+=1
    A=XtX+np.diag(lam_v)
    beta=np.linalg.solve(A,Xty)
    ratings={n:beta[s.idx[n]] for n in s.names}
    return sig, ratings, float(beta[-1])

# ------------------------------------ offense/defense split ridge (KenPom-ish)
def ridge_eff(s, lam=1.0, refit_every=8, tempo_lam=None):
    """Two ratings per team (offense, defense) on points-per-possession, plus a
    home term. Expected margin = (adjO_a - adjD_b) - (adjO_b - adjD_a) scaled by
    expected tempo. Walk-forward like ridge_margin."""
    nt=s.nt
    # possessions estimate: use torvik's per-game tempo (both teams share it)
    poss=s.tempo.copy()
    P=2*nt+1     # [off_0..off_n, def_0..def_n, home]
    XtX=np.zeros((P,P)); Xty=np.zeros(P)
    lam_v=np.full(P,lam); lam_v[-1]=1e-2
    sig=np.empty(s.n); beta=np.zeros(P); since=0
    mean_ppp=1.02
    for i in range(s.n):
        if since>=refit_every or i==0:
            beta=_solve(XtX+np.diag(lam_v),Xty); since=0
        a,b=s.a[i],s.b[i]
        h=0.0 if s.neutral[i] else 1.0
        # expected ppp for each side
        pa=mean_ppp+beta[a]+beta[nt+b]+h*beta[-1]
        pb=mean_ppp+beta[b]+beta[nt+a]-h*beta[-1]
        sig[i]=(pa-pb)*poss[i]
        # two observations per game
        ya=s.sa[i]/poss[i]-mean_ppp
        yb=s.sb[i]/poss[i]-mean_ppp
        for (o,d,yv,hh) in ((a,b,ya,h),(b,a,yb,-h)):
            j=[o,nt+d,P-1]; c=[1.0,1.0,hh]
            for u in range(3):
                for v in range(3):
                    XtX[j[u],j[v]]+=c[u]*c[v]
                Xty[j[u]]+=c[u]*yv
        since+=1
    beta=_solve(XtX+np.diag(lam_v),Xty)
    off={n:beta[s.idx[n]] for n in s.names}
    dff={n:beta[nt+s.idx[n]] for n in s.names}
    return sig, off, dff, float(beta[-1])

def ridge_eff_prior(s, lam=1.0, refit_every=24, prior_off=None, prior_def=None,
                    prior_w=0.0, regress=0.6, cap_ppp=None):
    """ridge_eff with pseudo-observations pulling each team toward `regress` times
    last season's offensive/defensive rating."""
    nt=s.nt; poss=s.tempo.copy(); P=2*nt+1
    XtX=np.zeros((P,P)); Xty=np.zeros(P)
    lam_v=np.full(P,lam); lam_v[-1]=1e-2
    if prior_w>0 and prior_off is not None:
        for name in s.names:
            j=s.idx[name]
            po=regress*prior_off.get(name,0.0); pd=regress*prior_def.get(name,0.0)
            XtX[j,j]+=prior_w; Xty[j]+=prior_w*po
            XtX[nt+j,nt+j]+=prior_w; Xty[nt+j]+=prior_w*pd
    sig=np.empty(s.n); beta=np.zeros(P); since=0; mean_ppp=1.02
    for i in range(s.n):
        if since>=refit_every or i==0:
            beta=_solve(XtX+np.diag(lam_v),Xty); since=0
        a,b=s.a[i],s.b[i]
        h=0.0 if s.neutral[i] else 1.0
        pa=mean_ppp+beta[a]+beta[nt+b]+h*beta[-1]
        pb=mean_ppp+beta[b]+beta[nt+a]-h*beta[-1]
        sig[i]=(pa-pb)*poss[i]
        ya=s.sa[i]/poss[i]-mean_ppp; yb=s.sb[i]/poss[i]-mean_ppp
        if cap_ppp:
            d=(ya-yb)/2.0; c=float(np.clip(d,-cap_ppp,cap_ppp)); mid=(ya+yb)/2.0
            ya,yb=mid+c,mid-c
        for (o,d_,yv,hh) in ((a,b,ya,h),(b,a,yb,-h)):
            j=[o,nt+d_,P-1]; c=[1.0,1.0,hh]
            for u in range(3):
                for v in range(3): XtX[j[u],j[v]]+=c[u]*c[v]
                Xty[j[u]]+=c[u]*yv
        since+=1
    beta=_solve(XtX+np.diag(lam_v),Xty)
    off={n:beta[s.idx[n]] for n in s.names}
    dff={n:beta[nt+s.idx[n]] for n in s.names}
    return sig, off, dff, float(beta[-1])
