"""Items 2-3 (Reviewer 1 Comments 2 and 8; Reviewer 2 Comment 1; Reviewer 3 Comment 2):
additional nonlinear ODEs and standard deterministic baselines.  y' = F(t, y), y(0) = y0, t in [0, T]."""
import numpy as np
from scipy.interpolate import BSpline
S2 = np.sqrt(2.0)
PROBLEMS = {
 'logistic-growth':  dict(F=lambda t, y: y*(1 - y), Fy=lambda t, y: 1 - 2*y, Ft=lambda t, y: 0*y, y0=0.5, T=1.0,
                          exact=lambda t: 1/(1 + np.exp(-t)), label='Logistic, $r=1$, $y_0=0.5$ (original growth)'),
 'logistic-decline': dict(F=lambda t, y: y*(1 - y), Fy=lambda t, y: 1 - 2*y, Ft=lambda t, y: 0*y, y0=2.0, T=1.0,
                          exact=lambda t: 1/(1 - 0.5*np.exp(-t)), label='Logistic, $r=1$, $y_0=2$ (original decline)'),
 'logistic-stiff':   dict(F=lambda t, y: 5*y*(1 - y), Fy=lambda t, y: 5*(1 - 2*y), Ft=lambda t, y: 0*y, y0=0.1, T=2.0,
                          exact=lambda t: 1/(1 + 9*np.exp(-5*t)), label='Logistic, $r=5$, $y_0=0.1$, $t\\in[0,2]$'),
 'gompertz':         dict(F=lambda t, y: -y*np.log(np.abs(y) + 1e-300), Fy=lambda t, y: -np.log(np.abs(y) + 1e-300) - 1, Ft=lambda t, y: 0*y,
                          y0=0.1, T=3.0, exact=lambda t: np.exp(np.log(0.1)*np.exp(-t)), label='Gompertz, $y\'=-y\\ln y$, $y_0=0.1$, $t\\in[0,3]$'),
 'riccati':          dict(F=lambda t, y: 1 + 2*y - y**2, Fy=lambda t, y: 2 - 2*y, Ft=lambda t, y: 0*y, y0=0.0, T=1.0,
                          exact=lambda t: 1 + S2*np.tanh(S2*t + 0.5*np.log((S2 - 1)/(S2 + 1))), label='Riccati, $y\'=1+2y-y^2$, $y_0=0$'),
}
def bspline_mats(n_seg, u, k=3):
    n = n_seg + k; kn = np.concatenate(([0.]*(k+1), np.linspace(0, 1, n_seg+1)[1:-1], [1.]*(k+1)))
    M = np.zeros((n, len(u))); dM = np.zeros((n, len(u))); d2M = np.zeros((n, len(u)))
    for i in range(n):
        e = np.zeros(n); e[i] = 1; s = BSpline(kn, e, k); M[i] = s(u); dM[i] = s.derivative(1)(u); d2M[i] = s.derivative(2)(u)
    return M, dM, d2M

class CurveODE:
    """Clamped cubic B-spline on [0,T] (n_seg=1: cubic Bezier); first control point fixed to y0; SSR of the ODE residual."""
    def __init__(self, name, n_seg, n_col=100):
        p = PROBLEMS[name]; self.p = p; self.T = p['T']; self.y0 = p['y0']; self.n_col = n_col
        self.tc = np.linspace(0, self.T, n_col); B, dB, _ = bspline_mats(n_seg, self.tc/self.T); self.B, self.dB = B, dB/self.T
        self.td = np.linspace(0, self.T, 1001); self.Bd = bspline_mats(n_seg, self.td/self.T)[0]; self.D = B.shape[0] - 1
    def resid(self, Z):
        Z = np.atleast_2d(Z); C = np.concatenate([np.full((len(Z), 1), self.y0), Z], 1); Y = C @ self.B
        return C @ self.dB - self.p['F'](self.tc, Y)
    def ssr(self, Z): return np.sum(self.resid(Z)**2, 1)
    def maxerr(self, z): return float(np.max(np.abs(np.r_[self.y0, z] @ self.Bd - self.p['exact'](self.td))))
