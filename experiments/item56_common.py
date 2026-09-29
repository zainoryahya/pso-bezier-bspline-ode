"""Shared definitions for the CMES revision experiments (Verhulst / logistic ODE, PSO with curve bases)."""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
r, K = 1.0, 1.0
tc = np.linspace(0, 1, 50); td = np.linspace(0, 1, 1001)
exact = lambda t, y0: K/(1.0 + ((K - y0)/y0)*np.exp(-r*t))

def bspline_mats(n_seg, t, k=3):
    n = n_seg + k                                   # number of control points
    kn = np.concatenate(([0.]*(k+1), np.linspace(0, 1, n_seg+1)[1:-1], [1.]*(k+1)))
    M = np.zeros((n, len(t))); dM = np.zeros((n, len(t)))
    for i in range(n):
        e = np.zeros(n); e[i] = 1; s = BSpline(kn, e, k); M[i] = s(t); dM[i] = s.derivative(1)(t)
    return M, dM

class Problem:
    """Clamped cubic B-spline with n_seg segments (n_seg=1 is the cubic Bezier curve); first control point = y0."""
    def __init__(self, n_seg, y0):
        self.y0 = y0; (self.B, self.dB) = bspline_mats(n_seg, tc); self.Bd = bspline_mats(n_seg, td)[0]; self.D = self.B.shape[0] - 1
    def resid(self, Z):
        Z = np.atleast_2d(Z); C = np.concatenate([np.full((len(Z), 1), self.y0), Z], 1); Y = C @ self.B
        return C @ self.dB - r*Y*(1 - Y/K)
    def ssr(self, Z): return np.sum(self.resid(Z)**2, 1)
    def maxerr(self, z): return float(np.max(np.abs(np.r_[self.y0, z] @ self.Bd - exact(td, self.y0))))
    def reference(self, n_starts=30, seed=99):
        rng = np.random.default_rng(seed); best = None
        for s in range(n_starts):
            rr = least_squares(lambda z: self.resid(z)[0], rng.uniform(-1, 3, self.D), method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
            if best is None or 2*rr.cost < best[1]: best = (rr.x, 2*rr.cost)
        return best

def pso(f, D, seed, lb=-1., ub=3., n_p=50, n_it=500, vclamp=0.2, scheme='constriction', stagnation=None):
    """PSO. scheme: 'constriction' (w=0.7298, c1=c2=1.49618) or 'inertia' (w: 0.9 -> 0.4 linearly, c1=c2=2.0).
    vclamp: fraction of the box width (None = no velocity limit). stagnation: stop when the global best has not
    improved by more than 1e-12 (relative) for this many iterations. Returns best, value, evaluations used."""
    rng = np.random.default_rng(seed); rg = ub - lb; vmax = None if vclamp is None else vclamp*rg
    pos = rng.uniform(lb, ub, (n_p, D)); v0 = rg if vmax is None else vmax
    vel = rng.uniform(-v0, v0, (n_p, D)); pb, pv = pos.copy(), f(pos); g = pb[pv.argmin()].copy(); gv = pv.min()
    evals = n_p; last_imp = 0
    for it in range(n_it):
        if scheme == 'constriction': w, c1, c2 = 0.7298, 1.49618, 1.49618
        else: w, c1, c2 = 0.9 - 0.5*it/max(n_it - 1, 1), 2.0, 2.0
        vel = w*vel + c1*rng.random((n_p, D))*(pb - pos) + c2*rng.random((n_p, D))*(g - pos)
        if vmax is not None: vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lb, ub); v = f(pos); evals += n_p
        i = v < pv; pb[i], pv[i] = pos[i], v[i]
        if pv.min() < gv:
            if gv - pv.min() > 1e-12*max(gv, 1e-300): last_imp = it
            gv = pv.min(); g = pb[pv.argmin()].copy()
        if stagnation and it - last_imp >= stagnation: break
    return g, float(gv), evals

def success_threshold(ssr_star): return max(1.05*ssr_star, ssr_star + 1e-14)
