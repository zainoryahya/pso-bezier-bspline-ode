"""
CMES 92459 revision, item 1 (Reviewer 2, Comment 2):
diagnose the 47-60% B-spline failure rate and test the fixes suggested by the reviewer.

Variants (all: Clerc-Kennedy constriction w=0.7298, c1=c2=1.49618, 50 particles x 500 iterations, 30 seeds):
  A  original : box [-5,10], no velocity limit, positions clipped             (as submitted)
  B  vclamp   : box [-5,10], |v| <= 0.2*(ub-lb)
  C  tight    : box [-1, 3], no velocity limit
  D  both     : box [-1, 3], |v| <= 0.2*(ub-lb)
  E  reset    : box [-5,10], no velocity limit, velocity set to 0 on the clipped components
Reference optimum of each basis: multi-start Levenberg-Marquardt (200 random starts in [-5,10]).
Success: final SSR <= 1.05 * SSR_ref.  Errors on a dense grid of 1001 points in [0,1].
"""
import numpy as np, json, time
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

r, K = 1.0, 1.0
REGIMES = [('growth', 0.5), ('equilibrium', 1.0), ('decline', 2.0)]
tc = np.linspace(0, 1, 50); td = np.linspace(0, 1, 1001)
exact = lambda t, y0: K/(1.0 + ((K - y0)/y0)*np.exp(-r*t))

def bern(t):
    return (np.stack([(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3]),
            np.stack([-3*(1-t)**2, 3*(1-t)**2 - 6*(1-t)*t, 6*(1-t)*t - 3*t**2, 3*t**2]))
def bsp(t, n=8, k=3):
    kn = np.concatenate(([0.]*(k+1), np.linspace(0, 1, n-k+1)[1:-1], [1.]*(k+1)))
    M = np.zeros((n, len(t))); dM = np.zeros((n, len(t)))
    for i in range(n):
        e = np.zeros(n); e[i] = 1; s = BSpline(kn, e, k); M[i] = s(t); dM[i] = s.derivative(1)(t)
    return M, dM
BASES = {'bezier': (bern(tc), bern(td)[0]), 'bspline': (bsp(tc), bsp(td)[0])}

def make(basis, y0):
    (B, dB), Bd = BASES[basis]
    def resid(Z):
        Z = np.atleast_2d(Z); C = np.concatenate([np.full((len(Z), 1), y0), Z], 1)
        Y = C @ B; dY = C @ dB; return dY - r*Y*(1 - Y/K)
    ssr = lambda Z: np.sum(resid(Z)**2, 1)
    def maxerr(z):
        C = np.concatenate([[y0], z]); return float(np.max(np.abs(C @ Bd - exact(td, y0))))
    return resid, ssr, maxerr, B.shape[0] - 1

def pso(f, D, seed, lb, ub, vclamp=False, reset=False, n_p=50, n_it=500, w=0.7298, c=1.49618):
    rng = np.random.default_rng(seed); rg = ub - lb; vmax = 0.2*rg
    pos = rng.uniform(lb, ub, (n_p, D))
    vel = rng.uniform(-vmax, vmax, (n_p, D)) if vclamp else rng.uniform(-rg, rg, (n_p, D))
    pb, pv = pos.copy(), f(pos); g = pb[pv.argmin()].copy(); gv = pv.min(); hits = 0
    for _ in range(n_it):
        vel = w*vel + c*rng.random((n_p, D))*(pb - pos) + c*rng.random((n_p, D))*(g - pos)
        if vclamp: vel = np.clip(vel, -vmax, vmax)
        new = pos + vel; out = (new < lb) | (new > ub); hits += out.sum()
        pos = np.clip(new, lb, ub)
        if reset: vel[out] = 0.0
        v = f(pos); i = v < pv; pb[i], pv[i] = pos[i], v[i]
        if pv.min() < gv: gv = pv.min(); g = pb[pv.argmin()].copy()
    return g, float(gv), hits/(n_it*n_p*D)

VARIANTS = {'A_original': dict(lb=-5., ub=10.), 'B_vclamp': dict(lb=-5., ub=10., vclamp=True),
            'C_tight': dict(lb=-1., ub=3.), 'D_both': dict(lb=-1., ub=3., vclamp=True),
            'E_reset': dict(lb=-5., ub=10., reset=True)}

def wilson(k, n, z=1.96):
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c - h), min(1, c + h)

out = {'reference': {}, 'runs': []}
for basis in BASES:
    for reg, y0 in REGIMES:
        resid, ssr, maxerr, D = make(basis, y0); rng = np.random.default_rng(99); best = None; finals = []
        t0 = time.process_time()
        for s in range(200):
            z0 = rng.uniform(-5, 10, D)
            rr = least_squares(lambda z: resid(z)[0], z0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
            finals.append(2*rr.cost)
            if best is None or 2*rr.cost < best[1]: best = (rr.x, 2*rr.cost)
        finals = np.array(finals); fs = best[1]
        J = np.array([(resid(best[0] + 1e-6*e)[0] - resid(best[0] - 1e-6*e)[0])/2e-6 for e in np.eye(D)]).T
        out['reference'][f'{basis}/{reg}'] = dict(D=D, ssr=fs, maxerr=maxerr(best[0]), z=best[0].tolist(),
            lm_multistart_success=float(np.mean(finals <= 1.05*fs + 1e-30)), lm_cpu_per_start=(time.process_time()-t0)/200,
            min_eig_JtJ=float(np.linalg.eigvalsh(J.T@J).min()), in_tight_box=bool(np.all((best[0] > -1) & (best[0] < 3))))
        print(basis, reg, 'ref SSR %.3e  maxerr %.3e  LM multistart %.0f%%  z range [%.2f, %.2f]' % (fs, maxerr(best[0]), 100*np.mean(finals <= 1.05*fs + 1e-30), best[0].min(), best[0].max()), flush=True)
        for vn, kw in VARIANTS.items():
            for s in range(30):
                seed = 1000*s + 7; t0 = time.perf_counter()
                z, v, hit = pso(ssr, D, seed, **kw)
                out['runs'].append(dict(basis=basis, regime=reg, variant=vn, seed=seed, ssr=v, maxerr=maxerr(z), z=z.tolist(),
                    success=bool(v <= 1.05*fs + 1e-30), time=time.perf_counter() - t0, bound_hit_frac=hit))
json.dump(out, open('item1_results.json', 'w'))
print('\nsuccess / median maxerr / mean bound-hit fraction')
for basis in BASES:
    for reg, _ in REGIMES:
        line = f'{basis:8} {reg:12}'
        for vn in VARIANTS:
            q = [r_ for r_ in out['runs'] if r_['basis'] == basis and r_['regime'] == reg and r_['variant'] == vn]
            k = sum(r_['success'] for r_ in q); lo, hi = wilson(k, 30)
            line += f" | {vn[:1]} {k:2d}/30 [{lo*100:.0f}-{hi*100:.0f}] {np.median([r_['maxerr'] for r_ in q]):.1e} hit {np.mean([r_['bound_hit_frac'] for r_ in q]):.2f}"
        print(line)
