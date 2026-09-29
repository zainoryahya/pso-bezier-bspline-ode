import json, numpy as np
BL = r'\\'
def sci(v):
    m, e = f'{v:.2e}'.split('e'); return '$' + m + r'\times10^{' + str(int(e)) + '}$'
NAMES = {'logistic-growth': 'Logistic growth', 'logistic-decline': 'Logistic decline', 'logistic-stiff': 'Logistic, $r=5$', 'gompertz': 'Gompertz', 'riccati': 'Riccati'}
A = json.load(open('itemA_optimum.json'))
L = [r'\begin{tabular}{llccccc}', r'\toprule', r'Basis & Regime & LM starts at optimum & Grid local minima & Hessian eigenvalues & cond$(H)$ & Max error' + BL, r'\midrule']
for k, d in A.items():
    b, r = k.split('/'); L.append(' & '.join(["B\\'ezier ($D=3$)" if b == 'bezier' else 'B-spline ($D=7$)', r, f"{d['lm_frac_at_opt']*1000:.0f}/1000",
        str(d.get('grid_local_minima', '--')) + (f" (of {d['grid_points']:,})".replace(',', '{,}') if 'grid_points' in d else ''), f"[{d['hess_eig_min']:.1f}, {d['hess_eig_max']:.1f}]", f"{d['hess_cond']:.1f}", sci(d['maxerr'])]) + BL)
L += [r'\bottomrule', r'\end{tabular}']; open('table_itemA_optimum.tex', 'w').write('\n'.join(L))
B = json.load(open('itemB_stats.json'))
L = [r'\begin{tabular}{llccccccc}', r'\toprule', r'Problem & Basis & Success [95\% CI] & Mean & SD & Median [95\% CI] & Best & Worst & CPU/run' + BL, r'\midrule']
for r in B['rows']:
    b = "B\\'ezier" if r['n_seg'] == 1 else f"{r['n_seg']}-seg."
    L.append(' & '.join([NAMES[r['problem']] if r['n_seg'] == 1 else '', b, f"{r['succ']}/30 [{r['succ_ci'][0]*100:.0f}, {r['succ_ci'][1]*100:.0f}]\\%", sci(r['mean']), sci(r['sd']),
        sci(r['median']) + ' [' + sci(r['median_ci'][0]) + ', ' + sci(r['median_ci'][1]) + ']', sci(r['best']), sci(r['worst']), f"{r['cpu_mean']:.2f} s"]) + BL)
    if r['n_seg'] == 10: L.append(r'\addlinespace')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}'); open('table_itemB_stats.tex', 'w').write('\n'.join(L))
L = [r'\begin{tabular}{llccc}', r'\toprule', r'Problem & Comparison & $p$ & $p$ (Holm) & Rank-biserial $r$' + BL, r'\midrule']
for t in B['tests']: L.append(' & '.join([NAMES[t['problem']], t['compare'], sci(t['p']), sci(t['p_holm']), f"{t['rank_biserial']:+.2f}"]) + BL)
L += [r'\bottomrule', r'\end{tabular}']; open('table_itemB_tests.tex', 'w').write('\n'.join(L))
C = json.load(open('itemC_minimax.json'))
L = [r'\begin{tabular}{llcccc}', r'\toprule', r'Basis / regime & Method & Success & Median $\max_k|R_k|$ & Median max error & CPU/run' + BL, r'\midrule']
for k, d in C.items():
    b, r = k.split('/'); first = True
    for m, v in d['methods'].items():
        L.append(' & '.join([(("B\\'ezier" if b == 'bezier' else 'B-spline') + f' / {r}') if first else '', m, f"{v['succ']}/30", sci(v['med_obj']), sci(v['med_err']), f"{v['cpu']:.2f} s"]) + BL); first = False
    L.append(r'\addlinespace')
L[-1] = r'\bottomrule'; L.append(r'\end{tabular}'); open('table_itemC_minimax.tex', 'w').write('\n'.join(L)); print('ok')
