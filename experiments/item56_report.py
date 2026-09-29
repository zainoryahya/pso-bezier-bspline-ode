import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S5 = json.load(open('item5_results.json'))['rows']; R6 = json.load(open('item6_results.json')); R6b = json.load(open('item6b_results.json'))
BL = r'\\'
def sci(v):
    m, e = f'{v:.2e}'.split('e'); return '$' + m + r'\times10^{' + str(int(e)) + '}$'
# ---- Table: sensitivity ----
L = [r'\begin{tabular}{llcccccc}', r'\toprule', r'Factor & Value & \multicolumn{3}{c}{Growth} & \multicolumn{3}{c}{Decline}' + BL,
     r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}', r' & & Success & Failures & Median error & Success & Failures & Median error' + BL, r'\midrule']
for fac in dict.fromkeys(r['factor'] for r in S5):
    for val in dict.fromkeys(r['value'] for r in S5 if r['factor'] == fac):
        g = next(r for r in S5 if r['factor'] == fac and r['value'] == val and r['regime'] == 'growth')
        d = next(r for r in S5 if r['factor'] == fac and r['value'] == val and r['regime'] == 'decline')
        L.append(' & '.join([fac, val.replace('None', 'none'), f"{g['succ']}/30", str(g['catastrophic']), sci(g['med_err']),
                             f"{d['succ']}/30", str(d['catastrophic']), sci(d['med_err'])]) + BL)
    L.append(r'\addlinespace')
L += [r'\bottomrule', r'\end{tabular}']; open('table_item5_sensitivity.tex', 'w').write('\n'.join(L))
# ---- Table: required iterations for >=27/30 success (50 particles) ----
def need(reg):
    out = {}
    for D in sorted({r['D'] for r in R6['rows']}):
        its = sorted([r for r in R6['rows'] if r['variant'] == 'corrected' and r['regime'] == reg and r['D'] == D], key=lambda r: r['n_it'])
        its_b = sorted([r for r in R6b if r['regime'] == reg and r['D'] == D and r['n_p'] == 50], key=lambda r: r['n_it'])
        cand = [r['n_it'] for r in its + its_b if r['succ'] >= 27]
        out[D] = min(cand) if cand else None
    return out
NG, ND = need('growth'), need('decline')
L = [r'\begin{tabular}{rccc}', r'\toprule', r'Free control points $D$ & Segments & \multicolumn{2}{c}{Iterations (50 particles) for $\geq$27/30 success}' + BL,
     r'\cmidrule(lr){3-4}', r' & & Growth & Decline' + BL, r'\midrule']
for D in NG:
    f = lambda v: '$\\leq125$' if v == 125 else ('$>16000$' if v is None else str(v))
    L.append(f'{D} & {D-2} & {f(NG[D])} & {f(ND[D])}' + BL)
L += [r'\bottomrule', r'\end{tabular}']; open('table_item6_budget.tex', 'w').write('\n'.join(L))
json.dump(dict(growth=NG, decline=ND), open('item6_required_iterations.json', 'w'), indent=1)
# fitted exponential rule on the points where the requirement is resolved (>125 and not None)
fits = {}
for reg, N in [('growth', NG), ('decline', ND)]:
    pts = [(D, v) for D, v in N.items() if v is not None and v > 125]
    b, a = np.polyfit([p[0] for p in pts], np.log2([p[1] for p in pts]), 1); fits[reg] = (a, b)
    print(reg, 'required iterations ~ 2^(%.2f + %.3f D)  -> doubling every %.1f control points' % (a, b, 1/b), pts)
json.dump({k: dict(intercept=v[0], slope=v[1], doubling_D=1/v[1]) for k, v in fits.items()}, open('item6_fit.json', 'w'), indent=1)
# ---- Figure: scalability ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3)); col = {125: '#c6dbef', 250: '#9ecae1', 500: '#6baed6', 1000: '#4292c6', 2000: '#2171b5', 4000: '#084594'}
for j, reg in enumerate(['growth', 'decline']):
    for it in [125, 250, 500, 1000, 2000, 4000]:
        rr = sorted([r for r in R6['rows'] if r['variant'] == 'corrected' and r['regime'] == reg and r['n_it'] == it], key=lambda r: r['D'])
        rs = sorted([r for r in R6['rows'] if r['variant'] == 'submitted' and r['regime'] == reg and r['n_it'] == it], key=lambda r: r['D'])
        ax[j].plot([r['D'] for r in rr], [r['succ'] for r in rr], '-o', color=col[it], ms=4, label=f'{it} it. (corrected)')
        if it in (500, 4000): ax[j].plot([r['D'] for r in rs], [r['succ'] for r in rs], '--x', color='#c05621' if it == 500 else '#7b341e', ms=5, label=f'{it} it. (submitted PSO)')
    ax[j].set(title=f'{reg} regime: success vs number of free control points', xlabel='free control points $D$', ylabel='successful runs (of 30)', ylim=(-1, 46), yticks=[0, 5, 10, 15, 20, 25, 30])
    ax[j].grid(alpha=.3); ax[j].legend(fontsize=6.5, ncol=2, loc='upper right', framealpha=.95)
for reg, c in [('growth', '#2b6cb0'), ('decline', '#c05621')]:
    N = NG if reg == 'growth' else ND; Ds = [D for D in N if N[D] is not None]
    ax[2].semilogy(Ds, [50*(N[D] + 1) for D in Ds], 'o-', color=c, label=f'{reg} (PSO, 50 particles)')
    a, b = fits[reg]; Dg = np.linspace(5, 22, 50); ax[2].semilogy(Dg, 50*2**(a + b*Dg), ':', color=c, lw=1)
ax[2].axhline(50*126, color='grey', lw=.6, ls='--'); ax[2].text(3.1, 50*126*1.15, 'smallest budget tested', fontsize=7, color='grey')
ax[2].set(title='evaluations needed for $\\geq$27/30 success', xlabel='free control points $D$', ylabel='objective evaluations'); ax[2].grid(alpha=.3, which='both'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_item6_scalability.png', dpi=200)
# ---- Figure: sensitivity ----
fig, ax = plt.subplots(1, 6, figsize=(16, 3.4), sharey=True)
for i, fac in enumerate(dict.fromkeys(r['factor'] for r in S5)):
    vals = list(dict.fromkeys(r['value'] for r in S5 if r['factor'] == fac)); x = np.arange(len(vals))
    for j, (reg, c) in enumerate([('growth', '#2b6cb0'), ('decline', '#c05621')]):
        k = [next(r for r in S5 if r['factor'] == fac and r['value'] == v and r['regime'] == reg)['succ'] for v in vals]
        ax[i].bar(x + (j - .5)*0.38, k, 0.38, color=c, label=reg)
    ax[i].set_xticks(x); ax[i].set_xticklabels([v.replace('None', 'none').replace('constriction', 'constr.').replace('(', '[').replace(')', ']').replace('.0', '') for v in vals], fontsize=7, rotation=30)
    ax[i].set_title(fac, fontsize=9); ax[i].grid(axis='y', alpha=.3)
ax[0].set_ylabel('successful runs (of 30)'); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('fig_item5_sensitivity.png', dpi=200)
print('ok')
