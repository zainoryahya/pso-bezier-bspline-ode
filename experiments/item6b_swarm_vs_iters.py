"""Item 6b: for the largest bases (D = 14, 17, 22), split a given evaluation budget between swarm size and iterations.
Budgets: 400k evaluations (50x8000, 100x4000, 200x2000) and 800k (50x16000, 100x8000, 200x4000). Corrected PSO, 30 seeds."""
import numpy as np, json
from multiprocessing import Pool
from item56_common import Problem, pso, success_threshold
REF = json.load(open('item6_results.json'))['ref']; REG = {'growth': 0.5, 'decline': 2.0}
CFG = [(50, 8000), (100, 4000), (200, 2000), (50, 16000), (100, 8000), (200, 4000)]
def job(a):
    ns, reg, (n_p, n_it) = a; P = Problem(ns, REG[reg]); fs, es = REF[f'{ns}/{reg}']; thr = success_threshold(fs)
    res = [pso(P.ssr, P.D, 1000*s + 7, n_p=n_p, n_it=n_it) for s in range(30)]
    e = np.array([P.maxerr(z) for z, _, _ in res])
    return dict(n_seg=ns, D=P.D, regime=reg, n_p=n_p, n_it=n_it, evals=n_p*(n_it + 1), succ=int(sum(f <= thr for _, f, _ in res)),
                med_ratio=float(np.median(e)/es), worst_ratio=float(e.max()/es))
if __name__ == '__main__':
    jobs = sorted([(ns, reg, c) for ns in [12, 15, 20] for reg in REG for c in CFG], key=lambda j: -j[2][0]*j[2][1])
    with Pool(2) as pool: rows = pool.map(job, jobs, chunksize=1)
    rows.sort(key=lambda r: (r['regime'], r['D'], r['evals'], r['n_p']))
    json.dump(rows, open('item6b_results.json', 'w'), indent=1)
    for r in rows: print(f"{r['regime']:8} D={r['D']:2d} {r['n_p']:3d}x{r['n_it']:5d} ({r['evals']/1e3:.0f}k)  succ {r['succ']:2d}/30  median err/err* {r['med_ratio']:.2f}  worst {r['worst_ratio']:.1f}")
