"""Reviewer 1 Comment 3 / Reviewer 3 Comment 7: numerical evidence for the reference optimum (original logistic problem).
(1) 1000 Levenberg-Marquardt starts, uniformly random in [-5,10]^D; (2) exhaustive grid scan of the Bezier SSR in 3-D;
(3) second-order check at the optimum (eigenvalues of the finite-difference Hessian of the SSR)."""
import numpy as np, json, time
from scipy.optimize import least_squares
from item56_common import Problem
out = {}
for ns, bname in [(1, 'bezier'), (5, 'bspline')]:
    for reg, y0 in [('growth', 0.5), ('decline', 2.0)]:
        P = Problem(ns, y0); rng = np.random.default_rng(2026); fin = []; X = []; t0 = time.process_time()
        for s in range(1000):
            rr = least_squares(lambda z: P.resid(z)[0], rng.uniform(-5, 10, P.D), method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
            fin.append(2*rr.cost); X.append(rr.x)
        fin = np.array(fin); X = np.array(X); i = int(fin.argmin()); zs = X[i]; fs = fin[i]
        spread = float(np.max(np.linalg.norm(X[fin <= 1.05*fs + 1e-14] - zs, axis=1)))
        h = 1e-4; E = np.eye(P.D); f = lambda z: P.ssr(z)[0]
        Hm = np.array([[(f(zs + h*E[a] + h*E[b]) - f(zs + h*E[a] - h*E[b]) - f(zs - h*E[a] + h*E[b]) + f(zs - h*E[a] - h*E[b]))/(4*h*h) for b in range(P.D)] for a in range(P.D)])
        ev = np.linalg.eigvalsh((Hm + Hm.T)/2)
        d = dict(D=P.D, ssr_star=float(fs), maxerr=P.maxerr(zs), lm_starts=1000, lm_frac_at_opt=float(np.mean(fin <= 1.05*fs + 1e-14)),
                 lm_distinct_final=int(len(np.unique(np.round(np.log10(fin + 1e-300), 3)))), max_dist_between_optimal_solutions=spread,
                 hess_eig_min=float(ev.min()), hess_eig_max=float(ev.max()), hess_cond=float(ev.max()/ev.min()), cpu=time.process_time() - t0)
        if ns == 1:   # exhaustive 3-D grid scan over [-1,3]^3 (121^3 points) and count of discrete local minima
            g = np.linspace(-1, 3, 121); G = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
            S = np.concatenate([P.ssr(G[k:k+200000]) for k in range(0, len(G), 200000)]).reshape(121, 121, 121)
            pad = np.pad(S, 1, constant_values=np.inf); core = pad[1:-1, 1:-1, 1:-1]; is_min = np.ones_like(S, bool)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx or dy or dz: is_min &= core <= pad[1+dx:122+dx, 1+dy:122+dy, 1+dz:122+dz]
            mins = np.argwhere(is_min)
            d.update(grid_points=int(S.size), grid_local_minima=int(len(mins)), grid_min_ssr=float(S.min()),
                     grid_min_location=[float(g[k]) for k in np.unravel_index(S.argmin(), S.shape)], grid_min_dist_to_opt=float(np.linalg.norm(np.array([g[k] for k in np.unravel_index(S.argmin(), S.shape)]) - zs)))
        out[f'{bname}/{reg}'] = d; print(bname, reg, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
json.dump(out, open('itemA_optimum.json', 'w'), indent=1)
