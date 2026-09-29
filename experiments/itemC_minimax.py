"""Reviewer 2 Comment 1(3): a non-smooth objective, max_k |R(t_k)| (minimax / L-infinity residual),
for which gradient-based least squares is not applicable. Original logistic problem (growth, decline),
Bezier (D=3) and 5-segment B-spline (D=7). 30 runs each: PSO (corrected), Nelder-Mead and BFGS (finite-difference
gradient) from uniformly random starts, and Nelder-Mead started from the least-squares (SSR) optimum.
Reference: the best value found by any method, refined by a linear-programming-free local polish (Nelder-Mead restarts)."""
import numpy as np, json, time, warnings
from scipy.optimize import minimize, least_squares
from item56_common import Problem, pso
warnings.filterwarnings('ignore')
out = {}
for ns, bn, it in [(1, 'bezier', 1000), (5, 'bspline', 4000)]:
    for reg, y0 in [('growth', 0.5), ('decline', 2.0)]:
        P = Problem(ns, y0); fmax = lambda Z: np.max(np.abs(P.resid(Z)), 1); f1 = lambda z: float(fmax(z)[0])
        ls = least_squares(lambda z: P.resid(z)[0], np.full(P.D, y0), method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15)
        res = {m: [] for m in ['PSO', 'Nelder-Mead', 'BFGS', 'NM from LSQ optimum']}; cpu = {m: [] for m in res}
        budget = 50*(it + 1)
        for s in range(30):
            seed = 1000*s + 7; rng = np.random.default_rng(seed + 1); z0 = rng.uniform(-5, 10, P.D)
            t0 = time.process_time(); z, f, _ = pso(fmax, P.D, seed, lb=-5., ub=10., n_p=50, n_it=it, vclamp=0.2); cpu['PSO'].append(time.process_time() - t0); res['PSO'].append((f, P.maxerr(z)))
            t0 = time.process_time(); r = minimize(f1, z0, method='Nelder-Mead', options=dict(maxfev=budget, xatol=1e-12, fatol=1e-14)); cpu['Nelder-Mead'].append(time.process_time() - t0); res['Nelder-Mead'].append((r.fun, P.maxerr(r.x)))
            t0 = time.process_time(); r = minimize(f1, z0, method='BFGS', options=dict(maxiter=budget)); cpu['BFGS'].append(time.process_time() - t0); res['BFGS'].append((r.fun, P.maxerr(r.x)))
            t0 = time.process_time(); r = minimize(f1, ls.x + 1e-3*rng.standard_normal(P.D), method='Nelder-Mead', options=dict(maxfev=budget, xatol=1e-12, fatol=1e-14)); cpu['NM from LSQ optimum'].append(time.process_time() - t0); res['NM from LSQ optimum'].append((r.fun, P.maxerr(r.x)))
        best = min(v[0] for m in res for v in res[m]); thr = 1.05*best
        d = dict(D=P.D, best_minimax=best, lsq_opt_minimax=f1(ls.x), lsq_opt_maxerr=P.maxerr(ls.x), methods={})
        for m in res:
            F = np.array([v[0] for v in res[m]]); E = np.array([v[1] for v in res[m]])
            d['methods'][m] = dict(succ=int(np.sum(F <= thr)), med_obj=float(np.median(F)), worst_obj=float(F.max()), med_err=float(np.median(E)), worst_err=float(E.max()), cpu=float(np.mean(cpu[m])))
        out[f'{bn}/{reg}'] = d
        print(f"\n{bn} {reg}: best max|R| {best:.3e} (least-squares optimum has max|R| {d['lsq_opt_minimax']:.3e}, max err {d['lsq_opt_maxerr']:.2e})")
        for m, v in d['methods'].items(): print(f"   {m:20} succ {v['succ']:2d}/30  median max|R| {v['med_obj']:.3e}  worst {v['worst_obj']:.2e}  median err {v['med_err']:.2e}  cpu {v['cpu']:.2f}s")
json.dump(out, open('itemC_minimax.json', 'w'), indent=1, default=float)
