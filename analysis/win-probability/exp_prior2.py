from carry import *
rows={'late':[],'early':[],'tourney':[]}
for pw,rg in [(0,0.7),(5,0.7),(15,0.7),(15,1.0),(40,0.7)]:
    n=f'ridgeEff prior_w{pw} r{rg}'
    r=score(chain_ridge_eff(lam=1.0,prior_w=pw,regress=rg,refit_every=40),n)
    for k in rows: rows[k].append(r[k])
    print(n,round(r['late']['logloss'],5),round(r['tourney']['logloss'],5),round(r['early']['logloss'],5),flush=True)
for k in rows:
    print();print('==',k,'==');print(fmt(rows[k]))
