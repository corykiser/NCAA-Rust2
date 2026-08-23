import numpy as np
from core import logloss
def fit_scale(sig, y, kind='logistic'):
    """1-D fit of signal -> probability. Returns (a,b) for p=1/(1+exp(-(a*sig+b)))."""
    X=np.column_stack([np.asarray(sig,dtype=float),np.ones(len(sig))])
    yv=np.asarray(y,dtype=float)
    w=np.zeros(2)
    for _ in range(200):
        z=X@w; p=1/(1+np.exp(-np.clip(z,-30,30)))
        g=X.T@(p-yv)
        W=p*(1-p)+1e-9
        H=X.T@(X*W[:,None])+1e-6*np.eye(2)
        step=np.linalg.solve(H,g)
        w-=step
        if np.max(np.abs(step))<1e-10: break
    return w
def apply_scale(sig,w):
    z=w[0]*np.asarray(sig,dtype=float)+w[1]
    return 1/(1+np.exp(-np.clip(z,-30,30)))
