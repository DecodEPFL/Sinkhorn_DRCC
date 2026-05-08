# Sinkhorn Distributionally Robust Controller

MATLAB and Python code to design and evaluate distributionally robust controllers for the AC5 (B-747 lateral dynamics) benchmark under Sinkhorn, Wasserstein, and nominal (empirical) uncertainty models.

Controllers are designed via System Level Synthesis (SLS) and tested on fresh i.i.d. noise realisations from the Dryden turbulence model.

---

## Files in this repository

### Benchmark
| File | Role |
|------|------|
| `COMPleib.m` | AC5 system matrices loader (COMPlib) |
| `COMPlib_r1_1/` | COMPlib data files |

### Noise model
| File | Role |
|------|------|
| `process_noise.m` | Dryden continuous turbulence model (B-747) |

### Controller design
| File | Role |
|------|------|
| `ac5_pilot_train_controllers.m` | Main training entry point |
| `golden_search.m` | Golden-section search for optimal dual variable λ |
| `nominal_trajectory_planning.m` | Nominal (empirical) SLS controller |
| `train_wasserstein_controller.m` | Wasserstein DRC controller |
| `directQCQP.m` | QCQP solver helper |

### i.i.d. out-of-sample testing
| File | Role |
|------|------|
| `ac5_iid_oos_test.m` | Evaluate all controllers on N fresh i.i.d. realisations |
| `ac5_iid_build_plot_data.m` | Export trajectory data (.mat) for the terminal-state plot |
| `export_ac5_plot_data.m` | Helper: package trajectories into .mat format |

### Plotting
| File | Role |
|------|------|
| `plot_ac5_iid_results.py` | Figure 1 (cost/violation bars) + Figure 2 (terminal-state scatter) |
| `plot_ac5_trajectories.py` | Stand-alone terminal-state scatter from a .mat file |

---

## Workflow

### 1. Train controllers

```matlab
ac5_pilot_train_controllers
```

Saves a controller bank to `experiment_outputs_full/`.

### 2. Run i.i.d. out-of-sample test

```matlab
ac5_iid_oos_test
```

Generates `N_test = 20 000` fresh noise realisations and evaluates every controller.  
Saves to `experiment_outputs_iid/`: results CSV, summary CSV, noise `.mat`.

### 3. Build plot data

```matlab
ac5_iid_build_plot_data
```

Picks the best Sinkhorn ε from the latest results CSV and exports full trajectories to `experiment_outputs_iid/ac5_iid_plot_data.mat`.  
Set `cfg.rho_target` and `cfg.n_plot` at the top of the script as needed.

### 4. Plot

```bash
python plot_ac5_iid_results.py
```

Saves to `experiment_outputs_iid/`:
- `iid_fig1_sinkhorn_improvement.pdf` — cost and violation-rate increment of Wasserstein and Empirical relative to best Sinkhorn, per ρ
- `iid_fig2_terminal_pairwise.pdf` — terminal-state scatter (β vs φ, p vs r) with terminal constraint box

---

## Dependencies

### Python

```
numpy, pandas, scipy, matplotlib
```

```bash
pip install -r requirements.txt
```

### MATLAB

- YALMIP with a compatible SDP solver (e.g. Mosek or SDPT3)
- Control System Toolbox (`c2d`, `lsim`, `tf`)

---

## Notes

- Generated `.mat` and `.csv` files are excluded from git; re-running the scripts reproduces them exactly.
- The Nominal controller is labelled *Empirical* in figures to match paper terminology.
- Random seed fixed to 42 in the test script (`rng(42)`) for reproducibility.
