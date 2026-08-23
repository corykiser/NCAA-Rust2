"""Walk-forward feature matrix: everything known before each game."""
import numpy as np, pickle
from core import *
import models

def build(s, lam=1.0, refit_every=16):
    nt=s.nt; poss=s.tempo.copy(); P=2*nt+1
    XtX=np.zeros((P,P)); Xty=np.zeros(P)
    lam_v=np.full(P,lam); lam_v[-1]=1e-2
    beta=np.zeros(P); since=0; mean_ppp=1.02
    # elo-margin state
    Rm=np.zeros(nt); Km=0.14
    # elo-win state
    Rw=np.full(nt,1500.0)
    # rolling season-to-date team stats
    gp=np.zeros(nt); last=np.full(nt,-99.0)
    sum_efg=np.zeros(nt); sum_tov=np.zeros(nt); sum_orb=np.zeros(nt); sum_ftr=np.zeros(nt)
    sum_defg=np.zeros(nt); sum_dtov=np.zeros(nt); sum_dorb=np.zeros(nt); sum_dftr=np.zeros(nt)
    sum_tempo=np.zeros(nt); sum_marg=np.zeros(nt); wins=np.zeros(nt)
    F=[];names=None
    for i in range(s.n):
        if since>=refit_every or i==0:
            beta=models._solve(XtX+np.diag(lam_v),Xty); since=0
        a,b=s.a[i],s.b[i]; h=0.0 if s.neutral[i] else 1.0
        oa,da,ob,db=beta[a],beta[nt+b],beta[b],beta[nt+a]
        pa=mean_ppp+oa+da+h*beta[-1]; pb=mean_ppp+ob+db-h*beta[-1]
        eff_sig=(pa-pb)*poss[i]
        ga,gb=max(gp[a],1),max(gp[b],1)
        row=dict(
            eff_margin=eff_sig,
            off_diff=beta[a]-beta[b], def_diff=beta[nt+b]-beta[nt+a],
            elo_margin=Rm[a]-Rm[b]+h*3.0,
            elo_win=(Rw[a]-Rw[b]+h*100.0)/100.0,
            neutral=1.0-h,
            gp_min=min(gp[a],gp[b]), gp_diff=gp[a]-gp[b],
            rest_a=min(s.day[i]-last[a],14) if last[a]>-90 else 7,
            rest_b=min(s.day[i]-last[b],14) if last[b]>-90 else 7,
            tempo_sum=(sum_tempo[a]/ga+sum_tempo[b]/gb),
            efg_diff=sum_efg[a]/ga-sum_efg[b]/gb,
            defg_diff=sum_defg[b]/gb-sum_defg[a]/ga,
            tov_diff=sum_tov[b]/gb-sum_tov[a]/ga,
            orb_diff=sum_orb[a]/ga-sum_orb[b]/gb,
            ftr_diff=sum_ftr[a]/ga-sum_ftr[b]/gb,
            marg_diff=sum_marg[a]/ga-sum_marg[b]/gb,
            wpct_diff=wins[a]/ga-wins[b]/gb,
            day=s.day[i],
        )
        if names is None: names=list(row.keys())
        F.append([row[k] for k in names])
        # ---- update states with the observed game
        r=s.rows[i]
        m=s.margin[i]; y=s.y[i]
        d=Rm[a]-Rm[b]+h*3.0
        err=m-d; Rm[a]+=0.5*Km*err; Rm[b]-=0.5*Km*err
        dw=Rw[a]-Rw[b]+h*100.0
        e=1/(1+10**(-dw/400)); wd=dw if y>0 else -dw
        mm=np.log(abs(m)+3)/np.log(3)*(2.2/(max(wd,0)*0.001+2.2) if wd>0 else 1.0)
        dd=20*mm*(y-e); Rw[a]+=dd; Rw[b]-=dd
        ya=s.sa[i]/poss[i]-mean_ppp; yb=s.sb[i]/poss[i]-mean_ppp
        for (o,dfn,yv,hh) in ((a,b,ya,h),(b,a,yb,-h)):
            j=[o,nt+dfn,P-1]; c=[1.0,1.0,hh]
            for u in range(3):
                for v in range(3): XtX[j[u],j[v]]+=c[u]*c[v]
                Xty[j[u]]+=c[u]*yv
        since+=1
        for (t,ef,tv,ob_,ft,de,dt,do,df) in ((a,r['efg'],r['tov'],r['orb'],r['ftr'],r['defg'],r['dtov'],r['dorb'],r['dftr']),
                                             (b,r['defg'],r['dtov'],r['dorb'],r['dftr'],r['efg'],r['tov'],r['orb'],r['ftr'])):
            def nz(x): return 0.0 if x!=x else x
            sum_efg[t]+=nz(ef);sum_tov[t]+=nz(tv);sum_orb[t]+=nz(ob_);sum_ftr[t]+=nz(ft)
            sum_defg[t]+=nz(de);sum_dtov[t]+=nz(dt);sum_dorb[t]+=nz(do);sum_dftr[t]+=nz(df)
            sum_tempo[t]+=poss[i]
        sum_marg[a]+=m; sum_marg[b]-=m; wins[a]+=y; wins[b]+=1-y
        gp[a]+=1; gp[b]+=1; last[a]=s.day[i]; last[b]=s.day[i]
    return np.array(F,dtype=float), names
