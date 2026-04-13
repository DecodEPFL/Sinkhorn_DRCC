# Sinkhorn Controller Design and Testing

This repository contains the MATLAB and Python code used to design and test robust controllers for the AC5 system under Sinkhorn, Wasserstein, and nominal uncertainty models.


## Workflow

### 1. Design controllers

Run the MATLAB training script to build Sinkhorn, Wasserstein, and nominal controllers.

Typical entry point:

```matlab
ac5_pilot_train_controllers
```

This trains controllers for the selected `rho_list` and saves a controller bank cache in the chosen output folder.

### 2. Generate testing distributions

Run the OOS testing script, which calls the Python generator to create adversarial distributions.

```matlab
ac5_pilot_test_oos
```

This produces testing distributions and evaluation CSVs in a separate output folder.

### 3. Plot results

To reproduce the comparison figure for best Sinkhorn vs Wasserstein vs nominal:

```bash
python plot_ac5_oos_results.py
```

To reproduce the terminal-state scatter plot for a single realization:

```bash
python plot_ac5_trajectories.py --input ac5_plot_data.mat --out-dir ac5_figures
```

## Dependencies

### Python

Typical packages used by the plotting and distribution-generation scripts:

- numpy
- pandas
- scipy
- matplotlib
- seaborn

You can pin these in a `requirements.txt` or `environment.yml` file.

### MATLAB

The MATLAB scripts assume:

- access to the AC5 benchmark setup
- the functions in this repository
- a working YALMIP / solver setup if your controller design scripts require it

## Notes

- Keep generated data outside of version control unless it is a curated example.
- If you want a cleaner repository, separate source code from outputs by using one folder for code and one ignored folder for all experiment artifacts.
- If you publish the repository, include one small example dataset and one example figure only if needed for documentation.

## Minimal GitHub checklist

If you only want the essentials, add:

- source scripts (`.m` and `.py`)
- `README.md`
- `.gitignore`
- dependency file for Python (`requirements.txt` or `environment.yml`)

