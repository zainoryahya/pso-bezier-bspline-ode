"""Reviewer 1 Comment 4: full statistics for all major configurations with the corrected PSO
(|v| <= 0.2*width, box [-5,10] for the new problems, [-1,3] for the original one), 30 seeds, dense-grid errors.
Reports mean, SD, median with 95% bootstrap CI, best, worst, success with 95% Wilson CI, and two-sided
Mann-Whitney U tests (Holm-corrected) between bases within each problem."""
import numpy as np, json, time
from multiprocessing import Pool
from scipy.stats import mannwhitneyu
from item23_problems import CurveODE, PROBLEMS
from item56_common import pso, success_threshold
REF = json.load(open('item23_results.json'))['ref']
CFG = [(1, 500), (5, 2000), (10, 8000)]
def job(a):
    name, ns, it = a; P = CurveODE(name, ns); thr = success_threshold(REF[f'{name}/{ns}']['ssr']); e = []; ok = []; tm = []
    for s in range(30):
        t0 = time.process_time(); z, f, _ = pso(P.ssr, P.D, 1000*s + 7, lb=-5., ub=10., n_p=50, n_it=it, vclamp=0.2)
        tm.append(time.process_time() - t0); e.append(P.maxerr(z)); ok.append(bool(f <= thr))
    return dict(problem=name, n_seg=ns, D=P.D, n_it=it, err=e, ok=ok, cpu=tm)
def wilson(k, n, z=1.96):
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d; return max(0, c-h), min(1, c+h)
def boot(x, B=5000):
    rng = np.random.default_rng(0); return np.percentile(np.median(rng.choice(x, (B, len(x))), 1), [2.5, 97.5])
if __name__ == '__main__':
    with Pool(2) as pool: runs = pool.map(job, sorted([(n, ns, it) for n in PROBLEMS for ns, it in CFG], key=lambda j: -j[2]), chunksize=1)
    rows = []; tests = []
    for n in PROBLEMS:
        rr = sorted([r for r in runs if r['problem'] == n], key=lambda r: r['n_seg'])
        for r in rr:
            e = np.array(r['err']); k = sum(r['ok']); lo, hi = wilson(k, 30); ci = boot(e)
            rows.append(dict(problem=n, n_seg=r['n_seg'], D=r['D'], n_it=r['n_it'], succ=k, succ_ci=[lo, hi], mean=e.mean(), sd=e.std(ddof=1), median=np.median(e),
                             median_ci=ci.tolist(), best=e.min(), worst=e.max(), cpu_mean=float(np.mean(r['cpu']))))
        pairs = [(0, 1), (0, 2), (1, 2)]; ps = [mannwhitneyu(rr[a]['err'], rr[b]['err'], alternative='two-sided').pvalue for a, b in pairs]
        order = np.argsort(ps); adj = [0]*3; run = 0
        for rank, i in enumerate(order): run = max(run, min(1, (3 - rank)*ps[i])); adj[i] = run
        for (a, b), p, pa in zip(pairs, ps, adj):
            A, B = np.array(rr[a]['err']), np.array(rr[b]['err']); U = mannwhitneyu(A, B).statistic; r_eff = 1 - 2*U/(len(A)*len(B))
            tests.append(dict(problem=n, compare=f"{rr[a]['n_seg']} vs {rr[b]['n_seg']} segments", p=p, p_holm=pa, rank_biserial=r_eff))
    json.dump(dict(rows=rows, tests=tests, runs=runs), open('itemB_stats.json', 'w'), indent=1, default=float)
    for r in rows: print(f"{r['problem']:17} {r['n_seg']:2d}seg succ {r['succ']:2d}/30 [{r['succ_ci'][0]*100:.0f},{r['succ_ci'][1]*100:.0f}] mean {r['mean']:.2e} sd {r['sd']:.2e} med {r['median']:.2e} [{r['median_ci'][0]:.1e},{r['median_ci'][1]:.1e}] best {r['best']:.2e} worst {r['worst']:.2e}")
    for t in tests: print(f"  {t['problem']:17} {t['compare']:18} p={t['p']:.1e} Holm={t['p_holm']:.1e} r={t['rank_biserial']:+.2f}")
