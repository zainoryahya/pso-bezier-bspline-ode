import json, numpy as np
from scipy.stats import mannwhitneyu
R = json.load(open('item1_results.json')); ref = R['reference']; runs = R['runs']
V = ['A_original', 'B_vclamp', 'C_tight', 'D_both', 'E_reset']
def wilson(k, n, z=1.96):
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d; return max(0, c-h), min(1, c+h)
def boot_ci_median(x, B=5000, seed=0):
    rng = np.random.default_rng(seed); m = np.median(rng.choice(x, (B, len(x))), 1); return np.percentile(m, [2.5, 97.5])
rows = []
for key, rf in ref.items():
    basis, reg = key.split('/'); fs = rf['ssr']; thr = max(1.05*fs, fs + 1e-12)
    for v in V:
        q = [r for r in runs if r['basis'] == basis and r['regime'] == reg and r['variant'] == v]
        e = np.array([r['maxerr'] for r in q]); s = np.array([r['ssr'] for r in q]); k = int(np.sum(s <= thr)); lo, hi = wilson(k, 30)
        ci = boot_ci_median(e)
        rows.append(dict(basis=basis, regime=reg, variant=v, succ=k, succ_lo=lo, succ_hi=hi, mean=e.mean(), sd=e.std(ddof=1), median=np.median(e),
                         med_ci=ci.tolist(), best=e.min(), worst=e.max(), time=np.mean([r['time'] for r in q]), hit=np.mean([r['bound_hit_frac'] for r in q])))
        print(f"{basis:7} {reg:11} {v:10} succ {k:2d}/30 [{lo*100:3.0f},{hi*100:3.0f}]  maxerr mean {e.mean():.2e} sd {e.std(ddof=1):.2e} med {np.median(e):.2e} [{ci[0]:.1e},{ci[1]:.1e}] best {e.min():.2e} worst {e.max():.2e}  t {rows[-1]['time']:.3f}s")
print('\nMann-Whitney U (max error), original A vs each fix, B-spline:')
tests = {}
for reg in ['growth', 'decline']:
    a = [r['maxerr'] for r in runs if r['basis'] == 'bspline' and r['regime'] == reg and r['variant'] == 'A_original']
    ps = []
    for v in V[1:]:
        b = [r['maxerr'] for r in runs if r['basis'] == 'bspline' and r['regime'] == reg and r['variant'] == v]
        ps.append((v, mannwhitneyu(a, b, alternative='two-sided').pvalue))
    # Holm correction
    order = np.argsort([p for _, p in ps]); m = len(ps); adj = [None]*m; run = 0
    for rank, i in enumerate(order): run = max(run, min(1, (m - rank)*ps[i][1])); adj[i] = run
    for (v, p), pa in zip(ps, adj): print(f'  {reg:8} A vs {v:10}: p = {p:.2e}, Holm-adjusted {pa:.2e}'); tests[f'{reg}/{v}'] = dict(p=p, p_holm=pa)
print('\nFailed original runs (B-spline): final SSR, max error, and control points at the box faces')
for reg in ['growth', 'decline']:
    for r in [r for r in runs if r['basis'] == 'bspline' and r['regime'] == reg and r['variant'] == 'A_original' and r['ssr'] > max(1.05*ref['bspline/'+reg]['ssr'], 1e-12)][:6]:
        z = np.array(r['z']); at = [f'C{i+1}={z[i]:.2f}' for i in range(len(z)) if abs(z[i] + 5) < 1e-9 or abs(z[i] - 10) < 1e-9]
        print(f'  {reg:8} seed {r["seed"]:5d}: SSR {r["ssr"]:.3e}, maxerr {r["maxerr"]:.3f}, at bounds: {at}')
json.dump(dict(rows=rows, tests=tests, reference=ref), open('item1_summary.json', 'w'), indent=1, default=float)
