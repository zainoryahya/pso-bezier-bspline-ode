import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from item23_problems import PROBLEMS
R = json.load(open('item23_results.json')); H = {(h['problem'], h['n_seg']): h for h in json.load(open('item23_hybrid.json'))}
BL = r'\\'
def sci(v):
    m, e = f'{v:.2e}'.split('e'); return '$' + m + r'\times10^{' + str(int(e)) + '}$'
NAMES = {'logistic-growth': 'Logistic growth ($r=1$)', 'logistic-decline': 'Logistic decline ($r=1$)', 'logistic-stiff': 'Logistic, $r=5$',
         'gompertz': 'Gompertz', 'riccati': 'Riccati'}
# Table A: PSO on all problems
L = [r'\begin{tabular}{llcccccc}', r'\toprule',
     r'Problem & Basis ($D$) & Basis optimum & PSO success & PSO median error & PSO worst error & LM random-start success & PSO+LM success' + BL, r'\midrule']
for n in PROBLEMS:
    for r in sorted([r for r in R['pso'] if r['problem'] == n], key=lambda r: r['n_seg']):
        b = 'B\\\'ezier' if r['n_seg'] == 1 else f"{r['n_seg']}-seg.\\ B-spline"
        h = H.get((n, r['n_seg'])); hs = f"{h['hybrid']}/30" if h else '--'
        L.append(' & '.join([NAMES[n] if r['n_seg'] == 1 else '', f"{b} ({r['D']})", sci(r['basis_opt_err']), f"{r['succ']}/30", sci(r['med_err']), sci(r['worst_err']),
                             f"{r['lm_success']*100:.0f}\\%", hs]) + BL)
    L.append(r'\addlinespace')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}'); open('table_item3_problems.tex', 'w').write('\n'.join(L))
# Table B: baselines vs PSO (5-seg and 10-seg) — error, F-evaluations, CPU
L = [r'\begin{tabular}{lllccc}', r'\toprule', r'Problem & Method & Setting & Max error (1001 pts) & $F$ evaluations & CPU time' + BL, r'\midrule']
def cpu(s): return f'{s*1e3:.2f} ms' if s < 1 else f'{s:.2f} s'
for n in PROBLEMS:
    first = True
    for b in [b for b in R['baselines'] if b['problem'] == n]:
        L.append(' & '.join([NAMES[n] if first else '', b['method'], b['param'], sci(b['maxerr']), str(b['f_evals']), cpu(b['cpu'])]) + BL); first = False
    for r in sorted([r for r in R['pso'] if r['problem'] == n and r['n_seg'] > 1], key=lambda r: r['n_seg']):
        L.append(' & '.join(['', 'PSO + B-spline (median of 30)', f"{r['n_seg']} seg., {r['n_it']} it.", sci(r['med_err']), f"{r['f_evals']:.1e}".replace('e+0', r'\times10^{').join(['$', '}$']) if False else sci(r['f_evals']), cpu(r['cpu'])]) + BL)
        rf = R['ref'][f"{n}/{r['n_seg']}"]
        L.append(' & '.join(['', 'LM + B-spline (one start)', f"{r['n_seg']} seg.", sci(rf['maxerr']), str(int(rf['lm_nfev']*100)), cpu(rf['lm_cpu'])]) + BL)
    L.append(r'\addlinespace')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}'); open('table_item2_baselines.tex', 'w').write('\n'.join(L))
# Figure: work-precision per problem
fig, ax = plt.subplots(1, 5, figsize=(19, 4.5), sharey=True)
st = {'RK4': ('#2b6cb0', 'o'), 'RK45': ('#4299e1', 's'), 'Chebyshev spectral': ('#2f855a', '^'), 'B-spline collocation + QM': ('#805ad5', 'D')}
for i, n in enumerate(PROBLEMS):
    for m, (c, mk) in st.items():
        bb = [b for b in R['baselines'] if b['problem'] == n and b['method'] == m]
        ax[i].loglog([b['f_evals'] for b in bb], [max(b['maxerr'], 1e-16) for b in bb], '-' + mk, color=c, label=m, ms=5)
    rr = sorted([r for r in R['pso'] if r['problem'] == n], key=lambda r: r['n_seg'])
    ax[i].loglog([r['f_evals'] for r in rr], [r['med_err'] for r in rr], '-o', color='#c05621', label='PSO + curve basis (median)', ms=6)
    ax[i].loglog([R['ref'][f"{n}/{r['n_seg']}"]['lm_nfev']*100 for r in rr], [R['ref'][f"{n}/{r['n_seg']}"]['maxerr'] for r in rr], '--x', color='#1a202c', label='LM + curve basis', ms=6)
    ax[i].set(title=NAMES[n].replace('$', '').replace('\\', ''), xlabel='right-hand-side evaluations $F(t,y)$'); ax[i].grid(alpha=.3, which='both')
ax[0].set_ylabel('max error on 1001 points'); ax[0].set_ylim(1e-16, 1e1)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=6, fontsize=9, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig('fig_item2_work_precision.png', dpi=180); print('ok')
