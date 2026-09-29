"""Deterministic baselines: RK4 (fixed step, cubic-Hermite dense output), RK45 (SciPy, adaptive),
Chebyshev spectral collocation (Newton), cubic B-spline collocation with quasilinearization (Newton)."""
import numpy as np, time
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, BarycentricInterpolator
from item23_problems import PROBLEMS, bspline_mats

def rk4(name, N):
    p = PROBLEMS[name]; F = p['F']; h = p['T']/N; t = np.linspace(0, p['T'], N + 1); y = np.zeros(N + 1); y[0] = p['y0']; nf = 0
    t0 = time.perf_counter()
    for i in range(N):
        k1 = F(t[i], y[i]); k2 = F(t[i] + h/2, y[i] + h/2*k1); k3 = F(t[i] + h/2, y[i] + h/2*k2); k4 = F(t[i] + h, y[i] + h*k3)
        y[i+1] = y[i] + h/6*(k1 + 2*k2 + 2*k3 + k4); nf += 4
    cpu = time.perf_counter() - t0; td = np.linspace(0, p['T'], 1001)
    yd = CubicHermiteSpline(t, y, F(t, y))(td)            # dense output (derivatives reuse available stage-1 values)
    return dict(method='RK4', param=f'N={N}', maxerr=float(np.max(np.abs(yd - p['exact'](td)))),
                maxerr_nodes=float(np.max(np.abs(y - p['exact'](t)))), f_evals=nf, cpu=cpu)

def rk45(name, rtol):
    p = PROBLEMS[name]; td = np.linspace(0, p['T'], 1001); t0 = time.perf_counter()
    s = solve_ivp(p['F'], (0, p['T']), [p['y0']], method='RK45', rtol=rtol, atol=rtol*1e-2, dense_output=True)
    cpu = time.perf_counter() - t0
    return dict(method='RK45', param=f'rtol={rtol:g}', maxerr=float(np.max(np.abs(s.sol(td)[0] - p['exact'](td)))), f_evals=int(s.nfev), cpu=cpu)

def chebyshev(name, N, tol=1e-13):
    p = PROBLEMS[name]; T = p['T']; F, Fy = p['F'], p['Fy']
    x = np.cos(np.pi*np.arange(N + 1)/N); c = np.r_[2, np.ones(N - 1), 2]*(-1)**np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T; dX = X - X.T; Dm = np.outer(c, 1/c)/(dX + np.eye(N + 1)); Dm -= np.diag(Dm.sum(1))
    t = (1 - x)*T/2; Dt = -2/T*Dm                       # t = 0 at x = 1 (index 0)
    y = np.full(N + 1, p['y0'], float); nf = 0; t0 = time.perf_counter()
    for it in range(50):
        G = Dt @ y - F(t, y); G[0] = y[0] - p['y0']; J = Dt - np.diag(Fy(t, y)); J[0] = 0; J[0, 0] = 1; nf += N + 1
        dy = np.linalg.solve(J, -G); y += dy
        if np.max(np.abs(dy)) < tol: break
    cpu = time.perf_counter() - t0; td = np.linspace(0, T, 1001)
    yd = BarycentricInterpolator(t, y)(td)
    return dict(method='Chebyshev spectral', param=f'N={N}', maxerr=float(np.max(np.abs(yd - p['exact'](td)))), f_evals=nf, newton_its=it + 1, cpu=cpu)

def bspline_qm(name, n_seg, tol=1e-13, c_init=None):
    """Cubic B-spline collocation at the n_seg+1 knots + initial condition + y''(0) condition; nonlinear system
    solved by quasilinearization (= Newton on the collocation equations)."""
    p = PROBLEMS[name]; T = p['T']; F, Fy, Ft = p['F'], p['Fy'], p['Ft']
    tk = np.linspace(0, T, n_seg + 1); B, dB, d2B = bspline_mats(n_seg, tk/T); dB /= T; d2B /= T**2
    y0 = p['y0']; ypp0 = float(Ft(0, y0) + Fy(0, y0)*F(0, y0))      # y''(0) from differentiating the ODE
    c = np.full(B.shape[0], y0, float) if c_init is None else c_init.copy(); nf = 0; t0 = time.perf_counter()
    for it in range(50):
        Y = c @ B; G = np.r_[c @ B[:, 0] - y0, c @ dB - F(tk, Y), c @ d2B[:, 0] - ypp0]
        J = np.vstack([B[:, 0], dB.T - (Fy(tk, Y)[:, None]*B.T), d2B[:, 0]]); nf += 2*(n_seg + 1)
        dc = np.linalg.solve(J, -G); c += dc
        if np.max(np.abs(dc)) < tol: break
    cpu = time.perf_counter() - t0; td = np.linspace(0, T, 1001); Bd = bspline_mats(n_seg, td/T)[0]
    return dict(method='B-spline collocation + QM', param=f'{n_seg} segments', maxerr=float(np.max(np.abs(c @ Bd - p['exact'](td)))),
                f_evals=nf, newton_its=it + 1, cpu=cpu, _c=c)

def bspline_qm_continuation(name, n_seg, start=5):
    """Mesh continuation: solve on `start` segments from a constant guess, then use each solution (resampled by
    least squares) as the initial guess on the doubled mesh, up to n_seg segments. Counts all evaluations."""
    p = PROBLEMS[name]; T = p['T']; ns = start; r = bspline_qm(name, ns); nf = r['f_evals']; cpu = r['cpu']
    while ns < n_seg:
        new = min(2*ns, n_seg); u = np.linspace(0, 1, 200)
        yprev = r['_c'] @ bspline_mats(ns, u)[0]; Bn = bspline_mats(new, u)[0]
        c0 = np.linalg.lstsq(Bn.T, yprev, rcond=None)[0]; r = bspline_qm(name, new, c_init=c0); nf += r['f_evals']; cpu += r['cpu']; ns = new
    r = dict(r); r['f_evals'] = nf; r['cpu'] = cpu; r['param'] = f'{n_seg} segments'; return r
