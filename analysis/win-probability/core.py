"""Common data structures + metrics for the win-probability shootout."""
import pickle, math, datetime
import numpy as np

YEARS=[2017,2018,2019,2021,2022,2023,2024,2025,2026]

class Season:
    """One season as flat arrays, chronological."""
    def __init__(self, year, rows):
        rows=sorted(rows,key=lambda r:(r['date'],r['team'],r['opp']))
        names=sorted({r['team'] for r in rows}|{r['opp'] for r in rows})
        self.year=year
        self.names=names
        self.idx={n:i for i,n in enumerate(names)}
        n=len(rows)
        self.n=n; self.nt=len(names)
        self.a=np.array([self.idx[r['team']] for r in rows])   # "home" side
        self.b=np.array([self.idx[r['opp']] for r in rows])
        self.sa=np.array([r['ts'] for r in rows],dtype=float)
        self.sb=np.array([r['os'] for r in rows],dtype=float)
        self.margin=self.sa-self.sb
        self.y=(self.margin>0).astype(float)
        self.neutral=np.array([r['site']=='N' for r in rows])
        self.tourney=np.array([r['tourney'] for r in rows])
        self.date=np.array([r['date'] for r in rows],dtype=object)
        self.day=np.array([(r['date']-rows[0]['date']).days for r in rows])
        self.tempo=np.array([r['tempo'] if r['tempo']==r['tempo'] else 68.0 for r in rows])
        self.conf=[r['conf'] for r in rows]; self.oppconf=[r['oppconf'] for r in rows]
        self.rows=rows
        # per-team conference (modal)
        self.team_conf={}
        for r in rows:
            self.team_conf.setdefault(r['team'],r['conf'])
            self.team_conf.setdefault(r['opp'],r['oppconf'])

def load_seasons(years=YEARS):
    raw=pickle.load(open('games.pkl','rb'))
    return {y:Season(y,raw[y]) for y in years}

EPS=1e-15
def logloss(p,y):
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS); y=np.asarray(y,dtype=float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def brier(p,y):
    return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def acc(p,y):
    p=np.asarray(p);y=np.asarray(y)
    return float(np.mean((p>0.5)==(y>0.5)))

def report(name, p, y, mask=None):
    if mask is not None: p,y=np.asarray(p)[mask],np.asarray(y)[mask]
    return dict(model=name,n=len(y),logloss=logloss(p,y),brier=brier(p,y),acc=acc(p,y))

def fmt(rs):
    hdr=f"{'model':<34}{'n':>7}{'logloss':>10}{'brier':>9}{'acc':>8}"
    out=[hdr,'-'*len(hdr)]
    for r in rs:
        out.append(f"{r['model']:<34}{r['n']:>7}{r['logloss']:>10.5f}{r['brier']:>9.5f}{r['acc']:>8.4f}")
    return "\n".join(out)
