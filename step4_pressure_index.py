"""
STEP 4 - The Pressure Index (PI).
Pressure is PERSPECTIVE-DEPENDENT: a batter is under pressure when the batting side is BEHIND and being squeezed,
a bowler is under pressure when the batting side is AHEAD and hitting out. Both are amplified when the match is on a knife-edge.

   PI_bat  = mean( Leverage , Collapse&Squeeze , Scoreboard-Strain-on-batting-side )
   PI_bowl = mean( Leverage , Runs-leaking (last 12 balls) , Scoreboard-Strain-on-bowling-side )

 Leverage         = how much the NEXT ball can swing win probability (step 2)
 Collapse&Squeeze = wickets + dot balls in the last 12 deliveries  (batting side being choked)
 Runs-leaking     = runs conceded in last 12 deliveries            (bowling side being hit)
 Strain           = 'shortfall' = what the batting side still needs (target, or par total in 1st inns)
                    minus what a typical team would add from this exact state (balls left, wickets lost)
Every part is turned into a within-innings percentile (0-1) so the parts are on one scale; parts are averaged.
'Decided' balls (win-prob outside 5%-95%) are neither pressure nor ordinary -> excluded (no garbage time).
High pressure = top 20% of PI among live balls (per innings); Ordinary = bottom 50%.
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor

def build_base():
    b = pd.read_pickle(str(OUT / 'balls_feat.pkl'))
    g = b.groupby(['match_id','innings'])
    b['runs_last12'] = g.team_runs.transform(lambda s: s.shift(1).rolling(12,min_periods=1).sum()).fillna(0)
    # expected remaining runs for a typical team from state (league, balls bowled, wickets lost): learned on innings 1 only
    tr = b[b.innings==1]; cols = ['is_ipl','legal_before','wkts_before']
    m = HistGradientBoostingRegressor(max_depth=4, learning_rate=.08, max_iter=150, min_samples_leaf=300,
                                      monotonic_cst=[0,-1,-1], random_state=0)
    m.fit(tr[cols], tr.inn1_total - tr.runs_before)
    b['E_rem'] = m.predict(b[cols])
    nxt = b[cols].copy(); nxt['wkts_before'] = np.minimum(nxt.wkts_before+1, 10)
    b['wkt_cost'] = np.maximum(b.E_rem - m.predict(nxt), 3.0)          # runs a team loses by losing one more wicket here
    par = b[b.innings==1].groupby('match_id').agg(t=('inn1_total','first'), ipl=('is_ipl','first')).groupby('ipl').t.mean()
    b['par_total'] = b.is_ipl.map(par)
    b['shortfall'] = np.where(b.innings==2, (b.target-b.runs_before) - b.E_rem, b.par_total - (b.runs_before+b.E_rem))
    b['phase'] = np.where(b.overid<=6,'PP',np.where(b.overid<=15,'MID','DEATH'))
    return b

def add_pi(b, w=(1,1,1), hi=0.20, lo=0.50, live=(0.05,0.95), by_phase=True):
    d = b.copy()
    d['live'] = d.wp_before.between(*live)
    key = ['innings','phase'] if by_phase else ['innings']      # rank within innings x phase => pressure is RELATIVE to the stage of the innings
    raw = pd.DataFrame({'innings':d.innings,'phase':d.phase,
        'lev':d.leverage,
        'mom_bat': 0.5*np.minimum(d.wk_last12,3)/3 + 0.5*d.dots_last12/12,
        'mom_bowl': d.runs_last12,
        'str_bat': d.shortfall, 'str_bowl': -d.shortfall})
    for c in ['lev','mom_bat','mom_bowl','str_bat','str_bowl']: raw[c] = raw[c].where(d.live)
    P = raw.groupby(key)[['lev','mom_bat','mom_bowl','str_bat','str_bowl']].rank(pct=True)
    s = float(sum(w))
    d['PI_bat']  = (w[0]*P.lev + w[1]*P.mom_bat  + w[2]*P.str_bat )/s
    d['PI_bowl'] = (w[0]*P.lev + w[1]*P.mom_bowl + w[2]*P.str_bowl)/s
    for k in ['bat','bowl']:
        col = f'PI_{k}'
        qh = d.groupby(key)[col].transform(lambda x: x.quantile(1-hi))
        ql = d.groupby(key)[col].transform(lambda x: x.quantile(lo))
        d[f'grp_{k}'] = np.where(~d.live,'decided', np.where(d[col]>=qh,'HIGH', np.where(d[col]<=ql,'ORD','mid')))
    return d

if __name__=='__main__':
    b = build_base()
    print('E_rem sanity: first ball of innings ~ par total ->', b[b.legal_before==0].groupby('is_ipl').E_rem.mean().round(1).to_dict(),
          '| par totals', b.groupby('is_ipl').par_total.first().round(1).to_dict())
    print('shortfall at ball 1 (should be ~0): inn1 %.1f  inn2 %.1f' % (b[(b.legal_before==0)&(b.innings==1)].shortfall.mean(), b[(b.legal_before==0)&(b.innings==2)].shortfall.mean()))
    print('mean wicket cost (runs):', b.wkt_cost.mean().round(1), ' early-over wicket:', b[b.legal_before<12].wkt_cost.mean().round(1), ' death wicket:', b[b.legal_before>=102].wkt_cost.mean().round(1))
    d = add_pi(b)
    d.to_pickle(str(OUT / 'balls_pi.pkl'))
    print('\nlive share of balls by innings:'); print(d.groupby('innings').live.mean().round(3).to_dict())
    for k in ['bat','bowl']:
        print(f'\n=== group sizes (PI_{k}) ==='); print(d.groupby(['innings',f'grp_{k}']).size().unstack().to_string())
        print(f'--- phase mix (% of balls) of HIGH vs ORD groups, PI_{k} ---')
        print((d[d[f'grp_{k}'].isin(['HIGH','ORD'])].groupby([f'grp_{k}','phase']).size().unstack().apply(lambda r: r/r.sum()*100,axis=1)).round(1).to_string())
        print(f'--- mean win-prob closeness, leverage, shortfall by group (PI_{k}) ---')
        print(d[d[f'grp_{k}'].isin(['HIGH','ORD'])].groupby(['innings',f'grp_{k}'])[['closeness','leverage','shortfall','wk_last12','dots_last12','runs_last12']].mean().round(3).to_string())
