"""
Run the whole pipeline in order.   Usage (from the SupeRR_Selector folder, venv active):
    python scripts/run_all.py            # full run (bootstrap = 200 resamples)
    python scripts/run_all.py 30         # quick run (bootstrap = 30 resamples)
"""
import subprocess, sys, time
from pathlib import Path
S = Path(__file__).resolve().parent
reps = sys.argv[1] if len(sys.argv) > 1 else '200'
STEPS = [('build_state.py', []), ('step2_winprob_leverage.py', []), ('step3_pressure_signals.py', []),
         ('step4_pressure_index.py', []), ('step5_ratings.py', []), ('step7_ensemble.py', []),
         ('step8_stability.py', ['variants']), ('step8_stability.py', ['boot', reps]), ('step9_final_tables.py', [])]
t0 = time.time()
for script, args in STEPS:
    t = time.time(); print(f'\n>>>>>> {script} {" ".join(args)}', flush=True)
    subprocess.run([sys.executable, str(S/script), *args], cwd=S, check=True)
    print(f'<<<<<< done in {time.time()-t:.0f}s', flush=True)
print(f'\nALL DONE in {(time.time()-t0)/60:.1f} min. Results are in the outputs/ folder.')
