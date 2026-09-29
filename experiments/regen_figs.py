"""Regenerate the single-run figures of the manuscript with the corrected PSO (seed 42, velocity limit, box [-1,3])."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from item56_common import Problem, exact, td
REG = [('growth', 0.5), ('equilibrium', 1.0), ('decline', 2.0)]; OUT = 'figures/'; import os; os.makedirs(OUT, exist_ok=True)
def pso_hist(P, seed=42, n_p=50, n_it=500, lb=-1., ub=3., w=0.7298, c=1.49618):
    rng = np.random.default_rng(seed); D = P.D; vmax = 0.2*(ub - lb)
    pos = rng.uniform(lb, ub, (n_p, D)); vel = rng.uniform(-vmax, vmax, (n_p, D)); pb, pv = pos.copy(), P.ssr(pos); g = pb[pv.argmin()].copy(); gv = pv.min(); h = [gv]
    for _ in range(n_it):
        vel = np.clip(w*vel + c*rng.random((n_p, D))*(pb - pos) + c*rng.random((n_p, D))*(g - pos), -vmax, vmax)
        pos = np.clip(pos + vel, lb, ub); v = P.ssr(pos); i = v < pv; pb[i], pv[i] = pos[i], v[i]
        if pv.min() < gv: gv = pv.min(); g = pb[pv.argmin()].copy()
        h.append(gv)
    return g, np.array(h)
res = {}
for ns, b in [(1, 'bezier'), (5, 'bspline')]:
    for reg, y0 in REG:
        P = Problem(ns, y0); z, h = pso_hist(P); Y = np.r_[y0, z] @ P.Bd; res[(b, reg)] = (Y, h, P.maxerr(z))
lab = {'bezier': 'PSO + cubic Bézier', 'bspline': 'PSO + cubic B-spline'}; col = {'bezier': '#c05621', 'bspline': '#2b6cb0'}
te = np.linspace(0.1, 1, 10)
for b in ['bezier', 'bspline']:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, (reg, y0) in enumerate(REG):
        Y, h, e = res[(b, reg)]; ax[j].plot(td, exact(td, y0), 'k-', lw=2.5, alpha=.35, label='exact')
        ax[j].plot(td, Y, '--', color=col[b], lw=1.5, label=lab[b]); ax[j].plot(te, np.interp(te, td, Y), 'o', color=col[b], ms=4)
        ax[j].set(title=f'{reg} ($y_0={y0}$)', xlabel='t', ylabel='y'); ax[j].grid(alpha=.3); ax[j].legend(fontsize=8)
        if reg == 'equilibrium': ax[j].set_ylim(0.9, 1.1)
    plt.tight_layout(); plt.savefig(OUT + f'solution_comparison_{b}.png', dpi=200); plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for j, (reg, y0) in enumerate(REG):
        Y, h, e = res[(b, reg)]; ax[j].semilogy(td, np.abs(Y - exact(td, y0)) + 1e-18, color=col[b], lw=1)
        ax[j].set(title=f'{reg}: max error {e:.2e}', xlabel='t', ylabel='|y - Y|'); ax[j].grid(alpha=.3, which='both')
    plt.tight_layout(); plt.savefig(OUT + f'absolute_error_{b}.png', dpi=200); plt.close()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for reg, c in zip(['growth', 'equilibrium', 'decline'], ['#2b6cb0', '#718096', '#c05621']):
        ax.semilogy(res[(b, reg)][1] + 1e-40, color=c, label=reg)
    ax.set(xlabel='iteration', ylabel='global-best SSR', title=f'{lab[b]}: convergence (seed 42)'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT + f'convergence_curve_{b}.png', dpi=200); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for j, (reg, y0) in enumerate(REG):
    ax[j].plot(td, exact(td, y0), 'k-', lw=2.5, alpha=.35, label='exact')
    for b in ['bezier', 'bspline']: ax[j].plot(td, res[(b, reg)][0], '--', color=col[b], lw=1.4, label=lab[b])
    ax[j].set(title=f'{reg} ($y_0={y0}$)', xlabel='t', ylabel='y'); ax[j].grid(alpha=.3); ax[j].legend(fontsize=8)
    if reg == 'equilibrium': ax[j].set_ylim(0.9, 1.1)
plt.tight_layout(); plt.savefig(OUT + 'comparison_bezier_bspline.png', dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(3)
for k, b in enumerate(['bezier', 'bspline']):
    ax.bar(x + (k - .5)*0.38, [max(res[(b, r)][2], 1e-17) for r, _ in REG], 0.38, color=col[b], label=lab[b])
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([r for r, _ in REG]); ax.set_ylabel('max error (1001 points)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.savefig(OUT + 'comparison_bar_chart.png', dpi=200); plt.close()
for k, v in res.items(): print(k, 'maxerr %.3e' % v[2], 'SSR@50 %.2e SSR@100 %.2e SSR@500 %.2e' % (v[1][50], v[1][100], v[1][-1]))
