"""
Try the chase win-probability model yourself: type a situation in, get a win probability out.
Run:  python3 step2b_try_the_model.py
"""
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUT = PROJECT_DIR / "outputs"
sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

b = pd.read_pickle(str(OUT / 'balls_state.pkl'))     # made by step1

def make_X(runs, wkts, legal, inn1_total, is_ipl):            # same features as step 2 (chase version)
    runs=np.asarray(runs,float); wkts=np.asarray(wkts,float); legal=np.asarray(legal,float)
    inn1_total=np.asarray(inn1_total,float); is_ipl=np.asarray(is_ipl,float)
    bl = 120-legal; crr = runs*6/np.maximum(legal,1)
    need = inn1_total+1-runs; rrr = need*6/np.maximum(bl,1); r = np.clip(rrr,0,40)
    X = {'runs':runs,'wkts':wkts,'balls_left':bl,'is_ipl':is_ipl,'crr':crr,'wb':(10-wkts)*bl,
         'need':need,'rrr':rrr,'inn1_total':inn1_total,'rrr_minus_crr':rrr-crr,'rrr_c2':r**2,
         'log_need':np.log1p(np.maximum(need,0)),'need_per_wkt':np.maximum(need,0)/np.maximum(10-wkts,1),'sqrt_bl':np.sqrt(bl)}
    X['need_per_wkt_sq']=X['need_per_wkt']**2
    return pd.DataFrame(X)

# ---- TRAIN: fit the formula's weights on every second-innings ball in the data ----
d = b[b.innings==2]
X = make_X(d.runs_after, d.wkts_after, d.legal_after, d.inn1_total, d.is_ipl)
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, d.bat_team_won)

# ---- ASK: describe a situation, get the chasing team's win probability ----
def chase_win_prob(target, runs_scored, wickets_lost, balls_left, ipl=True):
    X = make_X([runs_scored],[wickets_lost],[120-balls_left],[target-1],[int(ipl)])
    return model.predict_proba(X)[0,1]

print('Chasing 160, 30 needed off 18 balls, 3 wickets in hand (7 lost):')
print('   win prob = %.2f' % chase_win_prob(target=160, runs_scored=130, wickets_lost=7, balls_left=18))
print('Same, but 8 wickets in hand (2 lost):')
print('   win prob = %.2f' % chase_win_prob(target=160, runs_scored=130, wickets_lost=2, balls_left=18))
print('Chasing 160, 60 needed off 18 balls, 3 wickets in hand:')
print('   win prob = %.2f' % chase_win_prob(target=160, runs_scored=100, wickets_lost=7, balls_left=18))
