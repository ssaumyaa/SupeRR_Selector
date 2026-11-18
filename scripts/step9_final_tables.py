"""
STEP 9 - Merge the ensemble ranking with its bootstrap stability into one table: outputs/final_rankings.csv
"""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent.parent / "outputs"
parts = []
for role, lab in [('bat','Batter'), ('bowl','Bowler')]:
    e = pd.read_csv(OUT/f'ensemble_{role}.csv').head(10)
    bs = pd.read_csv(OUT/f'ensemble_bootstrap_{role}.csv')[['pid','P_top10_%','median_rank','rank_p10','rank_p90']]
    e = e.merge(bs, on='pid', how='left'); e.insert(0,'category',lab); parts.append(e)
f = pd.concat(parts).rename(columns={'score':'ensemble_PRS','share_of_defs_in_top10_%':'%definitions_in_top10',
    'nH':'high_pressure_balls','nO':'ordinary_balls','mH':'value_per_ball_high','mO':'value_per_ball_ordinary',
    'H_s':'pressure_level_shrunk','U_s':'resilience_shrunk','C_s':'consistency_shrunk','P_top10_%':'bootstrap_P_top10_%'})
f.round(3).to_csv(OUT/'final_rankings.csv', index=False)
print(f[['category','rank','name','ensemble_PRS','bootstrap_P_top10_%']].to_string(index=False))
