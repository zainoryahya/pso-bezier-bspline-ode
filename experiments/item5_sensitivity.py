"""Item 5 (Reviewer 1, Comment 6): one-factor-at-a-time hyperparameter sensitivity of the corrected PSO.
Base setting: 50 particles, 500 iterations, constriction coefficients, |v| <= 0.2*box width, box [-1,3].
Problem: 5-segment cubic B-spline (7 free control points), growth and decline regimes, 30 seeds per setting."""
import numpy as np, json
from multiprocessing import Pool
from item56_common import Problem, pso, success_threshold
BASE = dict(lb=-1., ub=3., n_p=50, n_it=500, vclamp=0.2, scheme='constriction', stagnation=None)
FACTORS = {
  'swarm size':      [('n_p', v) for v in [10, 20, 50, 100]],
  'iterations':      [('n_it', v) for v in [100, 250, 500, 1000]],
  'coefficients':    [('scheme', v) for v in ['constriction', 'inertia']],
  'velocity limit':  [('vclamp', v) for v in [None, 0.1, 0.2, 0.5]],
  'search box':      [('box', v) for v in [(-1., 3.), (-5., 10.), (-20., 20.)]],
  'stopping rule':   [('stagnation', v) for v in [None, 50, 100]],
}
REG = {'growth': 0.5, 'decline': 2.0}
def job(a):
    reg, fac, (k, v) = a; P = Problem(5, REG[reg]); zs, fs = REF[reg]; thr = success_threshold(fs)
    kw = dict(BASE)
    if k == 'box': kw['lb'], kw['ub'] = v
    else: kw[k] = v
    res = [pso(P.ssr, P.D, 1000*s + 7, **kw) for s in range(30)]
    e = np.array([P.maxerr(z) for z, _, _ in res]); ok = np.array([f <= thr for _, f, _ in res])
    return dict(regime=reg, factor=fac, param=k, value=str(v), succ=int(ok.sum()), med_err=float(np.median(e)), mean_err=float(e.mean()),
                sd_err=float(e.std(ddof=1)), worst_err=float(e.max()), catastrophic=int(np.sum(e > 1e-2)), evals=float(np.mean([n for _, _, n in res])))
REF = {}
if __name__ == '__main__':
    for reg, y0 in REG.items():
        P = Problem(5, y0); REF[reg] = P.reference()
    jobs = [(reg, fac, kv) for reg in REG for fac, lst in FACTORS.items() for kv in lst]
    with Pool(2) as pool: out = pool.map(job, jobs)
    json.dump(dict(base=BASE, ref={k: v[1] for k, v in REF.items()}, rows=out), open('item5_results.json', 'w'), indent=1)
    for r in out: print(f"{r['regime']:8} {r['factor']:15} {r['value']:14} succ {r['succ']:2d}/30 catastr {r['catastrophic']:2d}  med {r['med_err']:.2e}  worst {r['worst_err']:.2e}  evals {r['evals']:.0f}")
