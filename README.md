# PSO with cubic Bézier and B-spline bases for nonlinear growth equations

Code, raw results and figures for the manuscript

> Z. R. Yahya and S. Mat Zin, *Particle Swarm Optimization with Cubic Bézier and B-Spline Bases for Nonlinear Growth Equations: Reliability, Scalability, and Comparison with Classical Solvers*, submitted to *Computer Modeling in Engineering & Sciences* (CMES), manuscript 92459, revision 1.

The study treats PSO as a direct solver for first-order nonlinear ODEs, with the solution represented by a cubic Bézier curve or a clamped cubic B-spline. It covers the Verhulst (logistic) model in three regimes, plus a steep logistic, a Gompertz and a Riccati problem. The code includes:

- a diagnosis of PSO failures and the velocity-limit fix;
- numerical verification of the least-squares optimum;
- hyperparameter sensitivity and budget scaling;
- comparison with RK4, RK45, Chebyshev collocation and B-spline collocation;
- a non-smooth (minimax) objective.

## Contents

```
experiments/   all scripts, raw results (*.json) and generated LaTeX tables (*.tex)
figures/       the 12 figures of the manuscript
notebooks/     Google Colab notebook that re-runs the main experiments and checks the numbers
```

## Requirements

Python ≥ 3.10 with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`). The results in the paper were produced with Python 3.11, NumPy 2.4 and SciPy 1.17. All random seeds are fixed (`1000*s + 7`, s = 0,…,29), so re-running a script reproduces the stored results. With other library versions the last digits may differ, but success counts should not. CPU times depend on the machine and will differ from those in the paper.

Run every script from inside `experiments/`, because the scripts import one another and read and write files in that folder.

## Quick check (about 2 minutes)

```bash
cd experiments
python item1_pso_fix.py      # PSO variants A-E, 30 seeds each
python item1_analyse.py      # success rates, statistics  -> Table 8
python itemA_optimum.py      # 1000 LM starts, grid scan, Hessian -> Table 7
python regen_figs.py         # Figures 1-7 and dense-grid errors -> Table 6
python itemC_minimax.py      # minimax objective -> Table 16
```

Or open `notebooks/CMES_R1_reproduce_colab.ipynb` in Google Colab and choose *Runtime → Run all*. Its last cell compares 20 key values with the manuscript and prints PASS/FAIL for each.

The summary printed at the end of `item1_pso_fix.py` uses a purely relative success test (SSR ≤ 1.05·SSR\*), which is meaningless when SSR\* ≈ 0 (equilibrium regime). Use the output of `item1_analyse.py`, which applies the criterion used in the paper, SSR ≤ max(1.05·SSR\*, SSR\* + 10⁻¹⁴).

## Script → manuscript map

| Manuscript | Script(s) | Output | Run time* |
|---|---|---|---|
| Figs. 1–7, Table 6 | `regen_figs.py` | `figures/*.png` | seconds |
| Table 7 | `itemA_optimum.py` → `itemABC_tables.py` | `itemA_optimum.json`, `table_itemA_optimum.tex` | seconds |
| Table 8, Fig. 9 | `item1_pso_fix.py` → `item1_analyse.py` → `make_tables_extra.py`, `make_fig_item1.py` | `item1_*.json`, `table_item1_compact.tex`, `figures/fig_item1_success.png` | ~30 s |
| Table 9, Fig. 10 | `item5_sensitivity.py` → `item56_report.py` | `item5_results.json`, `table_item5_sensitivity.tex`, `fig_item5_sensitivity.png` | long |
| Table 10, Fig. 11 | `item6_scalability.py` → `item6b_swarm_vs_iters.py` → `item56_report.py` | `item6*.json`, `table_item6_budget.tex`, `fig_item6_scalability.png` | long |
| Table 11 | `itemD_regularized.py` → `make_tables_extra.py` | `itemD_regularized.json`, `table_itemD_regularized.tex` | long |
| Tables 12–14 | `item23_run.py` → `item23_hybrid.py` → `itemB_stats.py` → `item23_report.py`, `itemABC_tables.py` | `item23_*.json`, `itemB_stats.json`, `table_item3_problems.tex`, `table_itemB_*.tex` | long |
| Table 15, Fig. 12 | `item23_run.py` → `item23_report.py` → `make_tables_extra.py` | `table_item2_baselines.tex` (full), `table_item2_baselines_compact.tex` (as printed), `fig_item2_work_precision.png` | long |
| Table 16 | `itemC_minimax.py` → `itemABC_tables.py` | `itemC_minimax.json`, `table_itemC_minimax.tex` | ~40 s |

\*"long" means tens of minutes to several hours on a 2-core machine. The long scripts use `multiprocessing.Pool(2)`.

Shared code: `item56_common.py` (problem definitions, B-spline/Bézier bases, PSO with constriction and velocity limit, success criterion), `item23_problems.py` (additional ODEs) and `item23_baselines.py` (RK4, RK45, Chebyshev, B-spline collocation with quasilinearization).

## PSO settings used in the paper

Clerc–Kennedy constriction, w = 0.7298, c₁ = c₂ = 1.49618; velocity limit |v| ≤ 0.2 (ub − lb); search box [−1, 3] (Verhulst) or [−5, 10] (other problems); 50 particles; 30 independent seeds.

## Use of AI

The scripts were developed with the assistance of an AI coding assistant (Claude, Anthropic). The authors verified them and independently re-ran the main experiments in a separate environment (Google Colab), which reproduced all reported values.

## Licence

Code: MIT licence (see `LICENSE`). Results and figures: CC BY 4.0.

## How to cite

See `CITATION.cff`, or cite the article once it is published.
