import numpy as np, json, time
from multiprocessing import Pool
from scipy.optimize import least_squares
from item23_problems import PROBLEMS, CurveODE
from item23_baselines import rk4, rk45, chebyshev, bspline_qm_continuation
from item56_common import pso, success_threshold
CFG = [(1, 500), (5, 2000), (10, 8000)]            # (segments, PSO iterations with 50 particles)
def ref(name, ns, n_starts=50):
    P = CurveODE(name, ns); rng = np.random.default_rng(5); finals = []; best = None; t0 = time.process_time(); nfev = []
    for s in range(n_starts):
        rr = least_squares(lambda z: P.resid(z)[0], rng.uniform(-5, 10, P.D), method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
        finals.append(2*rr.cost); nfev.append(rr.nfev)
        if best is None or 2*rr.cost < best[1]: best = (rr.x, 2*rr.cost)
    finals = np.array(finals)
    return dict(ssr=best[1], maxerr=P.maxerr(best[0]), lm_success=float(np.mean(finals <= success_threshold(best[1]))),
                lm_cpu=(time.process_time() - t0)/n_starts, lm_nfev=float(np.mean(nfev)), D=P.D)
REF = {}
def init(r): REF.update(r)
def job(a):
    name, ns, it = a; P = CurveODE(name, ns); R = REF[f'{name}/{ns}']; thr = success_threshold(R['ssr']); out = []
    for s in range(30):
        t0 = time.process_time(); z, f, n = pso(P.ssr, P.D, 1000*s + 7, lb=-5., ub=10., n_p=50, n_it=it, vclamp=0.2)
        out.append((f <= thr, P.maxerr(z), time.process_time() - t0, n))
    e = np.array([o[1] for o in out])
    return dict(problem=name, n_seg=ns, D=P.D, n_it=it, succ=int(sum(o[0] for o in out)), med_err=float(np.median(e)), mean_err=float(e.mean()),
                sd_err=float(e.std(ddof=1)), best_err=float(e.min()), worst_err=float(e.max()), cpu=float(np.mean([o[2] for o in out])),
                obj_evals=int(out[0][3]), f_evals=int(out[0][3])*P.n_col, basis_opt_err=R['maxerr'], lm_success=R['lm_success'], lm_cpu=R['lm_cpu'])
if __name__ == '__main__':
    refs = {f'{n}/{ns}': ref(n, ns) for n in PROBLEMS for ns, _ in CFG}
    for k, v in refs.items(): print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()}, flush=True)
    with Pool(2, initializer=init, initargs=(refs,)) as pool:
        rows = pool.map(job, sorted([(n, ns, it) for n in PROBLEMS for ns, it in CFG], key=lambda j: -j[2]), chunksize=1)
    base = []
    for n in PROBLEMS:
        base += [rk4(n, N) for N in (10, 20, 40)] + [rk45(n, r) for r in (1e-6, 1e-10)] + [chebyshev(n, N) for N in (8, 16, 24)] + [bspline_qm_continuation(n, k) for k in (10, 20, 40)]
        for b in base[-11:]: b['problem'] = n
    for b in base: b.pop('_c', None)
    json.dump(dict(ref=refs, pso=rows, baselines=base), open('item23_results.json', 'w'), indent=1)
    for n in PROBLEMS:
        print('\n' + n)
        for r in [r for r in rows if r['problem'] == n]:
            print(f"  PSO {r['n_seg']:2d} seg (D={r['D']:2d}, {r['n_it']} it): succ {r['succ']:2d}/30 med {r['med_err']:.2e} worst {r['worst_err']:.2e} | basis opt {r['basis_opt_err']:.2e} | LM multistart {r['lm_success']*100:.0f}% {r['lm_cpu']*1e3:.1f} ms | PSO {r['cpu']:.2f} s, {r['f_evals']:.1e} F-evals")
        for b in [b for b in base if b['problem'] == n]:
            print(f"  {b['method']:26} {b['param']:12} maxerr {b['maxerr']:.2e}  F-evals {b['f_evals']:6d}  cpu {b['cpu']*1e3:.2f} ms")
