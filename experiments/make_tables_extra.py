"""Regenerate Table 8 (table_item1_compact.tex) and Table 11 (table_itemD_regularized.tex)
from item1_summary.json and itemD_regularized.json."""
import json, math

def sci(x):
    if x == 0:
        return r'$0$'
    e = math.floor(math.log10(abs(x))); m = x / 10**e
    if round(m, 2) >= 10:
        m /= 10; e += 1
    return rf'${m:.2f}\times10^{{{e}}}$'

# ---- Table 8: PSO variants, B-spline (D = 7) ----
S = {(r['basis'], r['regime'], r['variant']): r for r in json.load(open('item1_summary.json'))['rows']}
V = [('A_original', r'A: as submitted ($[-5,10]$, no velocity limit)'), ('B_vclamp', 'B: velocity limit only'),
     ('C_tight', r'C: box $[-1,3]$ only'), ('D_both', r'D: velocity limit + box $[-1,3]$ (adopted)'),
     ('E_reset', 'E: velocity reset at the bounds')]
L = [r'\begin{tabular}{lccccccccc}', r'\toprule',
     r'PSO variant & \multicolumn{3}{c}{Growth} & \multicolumn{3}{c}{Equilibrium} & \multicolumn{3}{c}{Decline}\\',
     r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
     r' & Success & Median & Worst & Success & Median & Worst & Success & Median & Worst\\', r'\midrule']
for v, name in V:
    cells = []
    for reg in ('growth', 'equilibrium', 'decline'):
        r = S['bspline', reg, v]; cells += [f"{r['succ']}/30", sci(r['median']), sci(r['worst'])]
    L.append(name + ' & ' + ' & '.join(cells) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open('table_item1_compact.tex', 'w').write('\n'.join(L))

# ---- Table 11: curvature-regularized objective, decline regime, 50 x 500 ----
D = [r for r in json.load(open('itemD_regularized.json')) if r['regime'] == 'decline' and r['n_p'] == 50 and r['n_it'] == 500]
lams = sorted({r['lam'] for r in D})
hdr = ['$\\lambda=0$'] + [f'$10^{{{int(round(math.log10(l)))}}}$' for l in lams if l > 0]
L = [r'\begin{tabular}{llcccccc}', r'\toprule', 'Control points & PSO & ' + ' & '.join(hdr) + r'\\', r'\midrule']
for ctrl in sorted({r['ctrl'] for r in D}):
    for k, (var, lab) in enumerate([('submitted', 'as submitted'), ('corrected', 'corrected')]):
        R = {r['lam']: r for r in D if r['ctrl'] == ctrl and r['variant'] == var}
        L.append((str(ctrl) if k == 0 else '') + f' & {lab} & ' + ' & '.join(f"{sci(R[l]['med'])} ({R[l]['n_fail']})" for l in lams) + r'\\')
L += [r'\bottomrule', r'\end{tabular}']
open('table_itemD_regularized.tex', 'w').write('\n'.join(L))
print('ok')

# ---- Table 15: compact baseline comparison (one representative setting per method) ----
def sci2(v):
    m, e = f'{v:.2e}'.split('e'); return '$' + m + r'\times10^{' + str(int(e)) + '}$'
def cpu(s): return f'{s*1e3:.2f} ms' if s < 1 else f'{s:.2f} s'
R = json.load(open('item23_results.json'))
NAMES = {'logistic-growth': 'Logistic growth', 'logistic-decline': 'Logistic decline', 'logistic-stiff': r'Logistic, $r=5$',
         'gompertz': 'Gompertz', 'riccati': 'Riccati'}
KEEP = {('RK4', 'N=40'), ('RK45', 'rtol=1e-10'), ('Chebyshev spectral', 'N=16'), ('B-spline collocation + QM', '20 segments')}
L = [r'\begin{tabular}{lllccc}', r'\toprule', r'Problem & Method & Setting & Max error (1001 pts) & $F$ evaluations & CPU time\\', r'\midrule']
for n in NAMES:
    first = True
    for b in [b for b in R['baselines'] if b['problem'] == n and (b['method'], b['param']) in KEEP]:
        L.append(' & '.join([NAMES[n] if first else '', b['method'], b['param'], sci2(b['maxerr']), str(b['f_evals']), cpu(b['cpu'])]) + r'\\'); first = False
    r = [r for r in R['pso'] if r['problem'] == n and r['n_seg'] == 10][0]
    L.append(' & '.join(['', 'PSO + B-spline (median of 30)', f"10 seg., 50$\\times${r['n_it']}", sci2(r['med_err']), sci2(r['f_evals']), cpu(r['cpu'])]) + r'\\')
    rf = R['ref'][f'{n}/10']
    L.append(' & '.join(['', 'LM + B-spline (one start)', '10 seg.', sci2(rf['maxerr']), str(int(rf['lm_nfev']*100)), cpu(rf['lm_cpu'])]) + r'\\')
    L.append(r'\addlinespace')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}')
open('table_item2_baselines_compact.tex', 'w').write('\n'.join(L))
print('ok compact')
