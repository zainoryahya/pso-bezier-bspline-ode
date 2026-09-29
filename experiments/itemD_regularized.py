"""Reviewer 2 Comment 3: rerun the regularized-SSR experiment (Section 4.3) with 30 seeds and the corrected PSO.
SSR_reg = SSR + lambda * sum_i Y''(t_i)^2 * dt (50 collocation points).  8, 12 and 16 control points (7, 11, 15 free),
decline regime (and growth for 8 points), standard budget 50 x 500; plus the enlarged 80 x 800 budget for 12 points."""
import numpy as np, json
from multiprocessing import Pool
from item56_common import Problem, pso, tc, bspline_mats
LAMS = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]; dt = tc[1] - tc[0]
def job(a):
    ns, reg, lam, n_p, n_it, variant = a; y0 = {'growth': 0.5, 'decline': 2.0}[reg]; P = Problem(ns, y0)
    d2 = np.zeros((ns + 3, len(tc)))
    from scipy.interpolate import BSpline
    k = 3; kn = np.concatenate(([0.]*4, np.linspace(0, 1, ns + 1)[1:-1], [1.]*4))
    for i in range(ns + 3):
        e = np.zeros(ns + 3); e[i] = 1; d2[i] = BSpline(kn, e, k).derivative(2)(tc)
    def f(Z):
        Z = np.atleast_2d(Z); C = np.concatenate([np.full((len(Z), 1), y0), Z], 1)
        return P.ssr(Z) + lam*np.sum((C @ d2)**2, 1)*dt
    kw = dict(lb=-1., ub=3., vclamp=0.2) if variant == 'corrected' else dict(lb=-5., ub=10., vclamp=None)
    e = np.array([P.maxerr(pso(f, P.D, 1000*s + 7, n_p=n_p, n_it=n_it, **kw)[0]) for s in range(30)])
    return dict(ctrl=ns + 3, D=P.D, regime=reg, lam=lam, n_p=n_p, n_it=n_it, variant=variant, med=float(np.median(e)), worst=float(e.max()),
                n_fail=int(np.sum(e > 1e-2)), q25=float(np.percentile(e, 25)), q75=float(np.percentile(e, 75)))
if __name__ == '__main__':
    jobs = [(ns, 'decline', l, 50, 500, v) for ns in (5, 9, 13) for l in LAMS for v in ('corrected', 'submitted')]
    jobs += [(5, 'growth', l, 50, 500, 'corrected') for l in (0, 1e-3)] + [(9, 'decline', 0, 80, 800, v) for v in ('corrected', 'submitted')]
    jobs += [(9, 'decline', 0, 50, 2000, 'corrected'), (13, 'decline', 0, 50, 2000, 'corrected')]
    with Pool(2) as pool: rows = pool.map(job, jobs)
    json.dump(rows, open('itemD_regularized.json', 'w'), indent=1)
    for r in rows: print(f"{r['variant']:9} {r['regime']:7} ctrl {r['ctrl']:2d} lam {r['lam']:7g} {r['n_p']}x{r['n_it']:4d}: median {r['med']:.2e} [IQR {r['q25']:.1e}-{r['q75']:.1e}]  fails(>1e-2) {r['n_fail']:2d}/30  worst {r['worst']:.2e}")
