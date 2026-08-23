import numpy as np, pickle, json
from core import *
from scipy_free import fit_scale, apply_scale
import feats
S=load_seasons(); BURNIN=45
X={};N=None
for y,s in S.items():
    X[y],N=feats.build(s)
    print('features',y,X[y].shape,flush=True)
pickle.dump((X,N),open('feats.pkl','wb'))

def loso_eval(name, predict):
    """predict(train_years, test_year) -> probabilities for test_year's games"""
    P={}
    for y in S: P[y]=predict(y)
    out={}
    for tag,mf in [('late',lambda s:s.day>=BURNIN),('tourney',lambda s:s.tourney)]:
        p=np.concatenate([P[y][mf(S[y])] for y in S]); yy=np.concatenate([S[y].y[mf(S[y])] for y in S])
        out[tag]=report(name,p,yy)
    return out

rows={'late':[],'tourney':[]}
# 1. single feature baselines through the logistic link
for col in ['eff_margin','elo_margin','elo_win']:
    j=N.index(col)
    def pr(y,j=j):
        tr=np.concatenate([X[z][:,j][S[z].day>=BURNIN] for z in S if z!=y])
        ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in S if z!=y])
        w=fit_scale(tr,ty); return apply_scale(X[y][:,j],w)
    r=loso_eval(f'link:{col}',pr)
    for k in rows: rows[k].append(r[k])
    print(col,'done',flush=True)

# 2. logistic regression on all features
def logistic_all(y, cols=None):
    cols=cols or list(range(len(N)))
    tr=np.concatenate([X[z][:,cols][S[z].day>=BURNIN] for z in S if z!=y])
    ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in S if z!=y])
    mu,sd=tr.mean(0),tr.std(0)+1e-9
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression(C=1.0,max_iter=2000).fit((tr-mu)/sd,ty)
    return lr.predict_proba((X[y][:,cols]-mu)/sd)[:,1]
r=loso_eval('logistic all feats',logistic_all)
for k in rows: rows[k].append(r[k]); 
print('logistic done',flush=True)

# 3. LightGBM
import lightgbm as lgb
def gbdt(y, **kw):
    cols=list(range(len(N)))
    trm=[z for z in S if z!=y]
    tr=np.concatenate([X[z][S[z].day>=BURNIN] for z in trm])
    ty=np.concatenate([S[z].y[S[z].day>=BURNIN] for z in trm])
    tw=np.concatenate([np.where(S[z].tourney[S[z].day>=BURNIN],6.0,1.0) for z in trm])
    m=lgb.LGBMClassifier(n_estimators=kw.get('n',600),learning_rate=kw.get('lr',0.03),
        num_leaves=kw.get('leaves',31),min_child_samples=kw.get('mcs',100),
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,verbose=-1)
    m.fit(tr,ty,sample_weight=tw)
    return m.predict_proba(X[y])[:,1]
for cfg in [dict(n=400,lr=0.03,leaves=15),dict(n=800,lr=0.02,leaves=31)]:
    r=loso_eval(f'lgbm {cfg}',lambda y,c=cfg: gbdt(y,**c))
    for k in rows: rows[k].append(r[k])
    print('lgbm',cfg,'done',flush=True)

for k in rows: print();print('==',k,'==');print(fmt(rows[k]))
