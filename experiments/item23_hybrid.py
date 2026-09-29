"""Hybrid check for the two hardest problems: polish each PSO result with Levenberg-Marquardt."""
import numpy as np, json
from multiprocessing import Pool
from scipy.optimize import least_squares
from item23_problems import CurveODE
from item56_common import pso, success_threshold
R = json.load(open('item23_results.json'))['ref']
def job(a):
    name, ns, it = a; P = CurveODE(name, ns); thr = success_threshold(R[f'{name}/{ns}']['ssr']); ok_p = ok_h = 0; eh = []
    for s in range(30):
        z, f, _ = pso(P.ssr, P.D, 1000*s + 7, lb=-5., ub=10., n_p=50, n_it=it, vclamp=0.2); ok_p += f <= thr
        rr = least_squares(lambda q: P.resid(q)[0], z, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        ok_h += 2*rr.cost <= thr; eh.append(P.maxerr(rr.x))
    return dict(problem=name, n_seg=ns, pso=int(ok_p), hybrid=int(ok_h), hybrid_med_err=float(np.median(eh)))
if __name__ == '__main__':
    with Pool(2) as pool: out = pool.map(job, [('logistic-stiff', 5, 2000), ('logistic-stiff', 10, 8000), ('gompertz', 5, 2000), ('gompertz', 10, 8000)])
    json.dump(out, open('item23_hybrid.json', 'w'), indent=1)
    for o in out: print(o)
