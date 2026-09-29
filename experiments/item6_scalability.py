"""Item 6 (Reviewer 1 Comment 7; Reviewer 3 Comments 4-5): systematic scalability study.
Cubic B-spline with n_seg segments (free control points D = n_seg + 2; n_seg = 1 is the cubic Bezier curve),
growth and decline regimes, 50 particles, iteration budgets 125 ... 4000, 30 seeds each.
Two PSO variants: corrected (|v| <= 0.2*width, box [-1,3]) and submitted (no velocity limit, box [-5,10])."""
import numpy as np, json, time
from multiprocessing import Pool
from item56_common import Problem, pso, success_threshold
SEGS = [1, 2, 3, 5, 7, 9, 12, 15, 20]; ITS = [125, 250, 500, 1000, 2000, 4000]; REG = {'growth': 0.5, 'decline': 2.0}
VAR = {'corrected': dict(lb=-1., ub=3., vclamp=0.2), 'submitted': dict(lb=-5., ub=10., vclamp=None)}
REF = {}
def init(ref): REF.update(ref)
def job(a):
    ns, reg, var, it = a; P = Problem(ns, REG[reg]); fs, es = REF[f'{ns}/{reg}']; thr = success_threshold(fs); out = []
    t0 = time.process_time()
    for s in range(30):
        z, f, n = pso(P.ssr, P.D, 1000*s + 7, n_p=50, n_it=it, **VAR[var]); out.append((f <= thr, P.maxerr(z)))
    ok = np.array([o for o, _ in out]); e = np.array([x for _, x in out])
    return dict(n_seg=ns, D=P.D, regime=reg, variant=var, n_it=it, evals=50*(it + 1), succ=int(ok.sum()),
                catastrophic=int(np.sum(e > 1e-2 + 10*es)), med_err=float(np.median(e)), worst_err=float(e.max()), err_star=es,
                cpu_per_run=(time.process_time() - t0)/30)
if __name__ == '__main__':
    ref = {}
    for ns in SEGS:
        for reg, y0 in REG.items():
            P = Problem(ns, y0); z, f = P.reference(); ref[f'{ns}/{reg}'] = (f, P.maxerr(z))
            print(ns, reg, 'D', P.D, 'SSR* %.2e err* %.2e' % (f, P.maxerr(z)), flush=True)
    jobs = sorted([(ns, reg, v, it) for ns in SEGS for reg in REG for v in VAR for it in ITS], key=lambda j: -j[3]*(j[0] + 3))
    with Pool(2, initializer=init, initargs=(ref,)) as pool: rows = pool.map(job, jobs, chunksize=1)
    rows.sort(key=lambda r: (r['variant'], r['regime'], r['n_seg'], r['n_it']))
    json.dump(dict(ref=ref, rows=rows), open('item6_results.json', 'w'), indent=1)
    for var in VAR:
        for reg in REG:
            print(f'\n{var} / {reg}: successes out of 30 (rows: D, columns: iterations {ITS})')
            for ns in SEGS:
                rr = [r for r in rows if r['variant'] == var and r['regime'] == reg and r['n_seg'] == ns]
                print(f"  D={rr[0]['D']:2d} err*={rr[0]['err_star']:.1e} | " + ' '.join(f"{r['succ']:2d}({r['catastrophic']:2d})" for r in rr))
