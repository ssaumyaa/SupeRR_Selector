"""
STEP 3 - Build candidate PRESSURE SIGNALS (all computed from info available BEFORE the ball is bowled)
and test which ones actually change batting outcomes, controlling for phase / league / batting order.
Output: balls_feat.pkl  + evidence table (printed)
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')


def ols_cluster(y, X, groups):
    """OLS with match-clustered standard errors (same small-sample correction statsmodels uses). Returns coefficients and t-stats."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    g = pd.factorize(groups)[0]; G = g.max() + 1
    S = pd.DataFrame(X * resid[:, None]).groupby(g).sum().values      # per-match sums of x_i * e_i
    V = (G/(G-1)) * ((n-1)/(n-k)) * XtX_inv @ (S.T @ S) @ XtX_inv
    return beta, beta/np.sqrt(np.diag(V))

b = pd.read_pickle(str(OUT / 'balls_lev.pkl')).sort_values(['match_id','innings','innings_ball_num']).reset_index(drop=True)
b['is_dot'] = ((b.bat_runs==0)&(b.team_runs==0)&b.is_legal).astype(int)
b['bnd']    = (b.is_four|b.is_six).astype(int)
g = b.groupby(['match_id','innings'])
# momentum signals: last 12 deliveries BEFORE this one
b['wk_last12']   = g.wkt.transform(lambda s: s.shift(1).rolling(12,min_periods=1).sum()).fillna(0)
b['dots_last12'] = g.is_dot.transform(lambda s: s.shift(1).rolling(12,min_periods=1).sum()).fillna(0)
b['striker_balls'] = b.groupby(['match_id','innings','striker_id']).cumcount()
b['new_batter'] = (b.striker_balls<=5).astype(int)           # settling-in effect -> a CONTROL, not 'pressure'
# chase signals
b['crr']  = b.runs_before*6/np.maximum(b.legal_before,1)
b['need'] = b.target-b.runs_before
b['rrr']  = b.need*6/np.maximum(b.balls_left_before,1)
b['rrr_gap']   = (b.rrr-b.crr).clip(-10,15)                   # how far behind the asking rate
b['closeness'] = 1-np.abs(2*b.wp_before-1)                    # 1 = coin-flip match, 0 = decided
b['wkts_down'] = b.wkts_before
b['death'] = (b.overid>=16).astype(int); b['pp'] = (b.overid<=6).astype(int)
b.to_pickle(str(OUT / 'balls_feat.pkl'))

d = b[b.is_legal].copy(); res=[]
spec = {1:['wk_last12','dots_last12','leverage','new_batter','wkts_down','knock_out'],
        2:['wk_last12','dots_last12','leverage','closeness','rrr_gap','new_batter','wkts_down','knock_out']}
for inn,sigs in spec.items():
    e = d[d.innings==inn]
    for target in ['bat_runs','wkt']:
        X = e[sigs+['death','pp','is_ipl','batting_order']].astype(float)
        Xz = (X-X.mean())/X.std(); Xz.insert(0, 'const', 1.0)
        beta, t = ols_cluster(e[target].astype(float).values, Xz.values, e.match_id.values)
        out = pd.DataFrame({'coef':beta,'t':t}, index=Xz.columns).loc[sigs]; out['inn']=inn; out['y']=target; res.append(out)
R = pd.concat(res).reset_index().rename(columns={'index':'signal'})
print('Effect of +1 SD of each signal on runs-per-ball and wicket-probability (t-stats are cluster-robust by match)')
print(R.pivot_table(index=['inn','signal'],columns='y',values=['coef','t']).round(3).to_string())
