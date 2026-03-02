# Staged Calibration Workflow

## Rationale

The chimney model has ~10 tunable parameters spanning momentum (leak area,
discharge coefficient, minor-loss K), heat transfer (convection multipliers,
emissivity), wind effects (cap Cp), and door geometry.  Fitting all parameters
simultaneously is ill-conditioned because several knobs trade off against each
other (e.g., A_leak vs Cd both affect flow resistance).

The staged approach fixes parameters in order of decreasing identifiability:

| Stage | Parameters | Targets | Tests |
|-------|-----------|---------|-------|
| **A** | A_leak_cm2, Cd, K_stove, K_minor_mult | SS mass flow + pressure profile (closed door) | All closed-door |
| **B** | h_inner_mult, h_outer_mult, eps_inner_scale | Temperature RMSE (closed door) | All closed-door |
| **C** | Cp_override | SS mass flow vs wind speed | Cross-wind tests |
| **D** | door_area_scale, door_ramp_s | Cracked-door mass flow | Door-transition tests |

Each stage freezes the parameters from all prior stages.

## Code layout

```
flued_module/calibration/
    __init__.py          # public exports
    param_space.py       # StageAParams .. StageDParams dataclasses + LHS sampling
    objectives.py        # compute_stage_A/B/C_objective()
    calibrator.py        # StagedCalibrator (sweep, refine, cache, save)

notebooks/
    calibration_staged.ipynb   # run notebook (all stages)
```

## How to run

1. Activate the `chimney_ht` conda environment.
2. Open `notebooks/calibration_staged.ipynb` in Jupyter.
3. In the **Config** cell, set sample counts:
   - Smoke test: `N_COARSE_A = 5`, `N_COARSE_B = 5`
   - Production: `N_COARSE_A = 60`, `N_COARSE_B = 40`
4. Run all cells.  Stage A takes ~2 min per sample; Stage B is similar.
5. Outputs are saved to `outputs/calibration/<timestamp>/`.

## Outputs

Each stage directory contains:

- `cache.json` -- evaluated parameter sets (survives restarts)
- `results.json` -- full sweep results
- `best_params.json` -- best-fit parameter set for that stage

The `final_results_plots/` directory contains:

- `fig_mdot_comparison.png` -- bar chart: sim vs exp steady-state mass flow
- `fig_temp_rmse.png` -- bar chart: per-sensor temperature RMSE
- Per-test overlay plots (temperature, pressure)

A markdown report cell at the bottom prints a summary between
`BEGIN_CALIBRATION_REPORT` / `END_CALIBRATION_REPORT` delimiters.

## Search algorithm

- **Coarse sweep**: Latin hypercube sampling over the parameter bounds.
  Deterministic with `RNG_SEED = 42`.
- **Local refinement**: Nelder-Mead simplex on the top candidate from the
  coarse sweep (maxiter = 40, clipped to bounds).
- **Caching**: Every evaluation is keyed by MD5 hash of the parameter dict.
  Re-running the notebook skips cached evaluations.

## Guardrails

- All parameters are clipped to physically motivated bounds (see
  `param_space.py` for exact values).
- `validate()` on each dataclass rejects out-of-bounds samples.
- Objective functions use normalized residuals (divide by experimental
  uncertainty or a fixed sigma scale) so that mdot and pressure terms are
  comparable.
- Failed simulations return objective = 1e12 (effectively excluded).

## Key constraints

- The calibration harness does **not** modify solver equations.  It only
  adjusts input parameters (config dicts, multipliers set on the constructed
  chimney object) and evaluates the existing `run_simulation()` method.
- Stages C and D are optional and disabled by default (set
  `RUN_STAGE_C = True` / `RUN_STAGE_D = True` to enable).
