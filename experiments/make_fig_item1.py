"""Regenerate Figure 9 (fig_item1_success.png) from item1_summary.json."""
import json, os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = {(r['basis'], r['regime'], r['variant']): r for r in json.load(open('item1_summary.json'))['rows']}
V = ['A_original', 'B_vclamp', 'C_tight', 'D_both', 'E_reset']
LAB = ['A: submitted\n[−5,10], no v-limit', 'B: velocity\nclamp', 'C: box\n[−1,3]', 'D: clamp +\nbox [−1,3]', 'E: velocity\nreset']
REG = [('growth', '#2b6cb0'), ('equilibrium', '#718096'), ('decline', '#c05621')]
fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.26; x = np.arange(len(V))
for k, (reg, col) in enumerate(REG):
    r = [S['bspline', reg, v] for v in V]; y = np.array([q['succ'] for q in r])
    lo = y - 30*np.array([q['succ_lo'] for q in r]); hi = 30*np.array([q['succ_hi'] for q in r]) - y
    ax.bar(x + (k-1)*w, y, w, color=col, label=reg, yerr=[lo, hi], capsize=3, error_kw=dict(lw=0.9))
ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=8.5); ax.set_ylim(0, 36); ax.set_ylabel('successful runs (of 30)')
ax.set_title('Cubic B-spline (7 free control points): PSO success over 30 seeds, 95% Wilson intervals', fontsize=10)
ax.grid(axis='y', alpha=0.3); ax.legend(ncol=3, loc='upper center', fontsize=8, frameon=False)
os.makedirs('figures', exist_ok=True); plt.tight_layout(); plt.savefig('figures/fig_item1_success.png', dpi=200); print('ok')
