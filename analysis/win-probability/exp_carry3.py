from carry import *
res=[]
for rg in [0.5,0.7,0.8,0.9,1.0]:
    for K in [0.06,0.10,0.14,0.20]:
        n=f'eloM K{K} r{rg}'
        r=score(chain_elo_margin(K=K,regress=rg),n)
        res.append((r['late']['logloss'],r))
        print(n,round(r['late']['logloss'],5),round(r['tourney']['logloss'],5),round(r['early']['logloss'],5),flush=True)
res.sort(key=lambda x:x[0])
for k in ['late','tourney','early']:
    print();print('==',k,'==');print(fmt([r[k] for _,r in res[:10]]))
