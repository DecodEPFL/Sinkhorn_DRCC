import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (18, 7)
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 19
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 17
plt.rcParams["legend.fontsize"] = 17
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUTPUT_DIR = Path("experiment_outputs_full")


def configure_plot_fonts(use_tex=True):
    has_latex = shutil.which("latex") is not None
    if use_tex and has_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times"],
                "axes.unicode_minus": False,
            }
        )
        print("Using full LaTeX rendering with Times New Roman (text.usetex=True).")
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "mathtext.fontset": "cm",
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "axes.unicode_minus": False,
            }
        )
        if use_tex:
            print("LaTeX binary not found. Falling back to Times Roman with mathtext.")


def fmt_sci_tex(val):
    exp = int(np.floor(np.log10(val)))
    coeff = val / (10 ** exp)
    if np.isclose(coeff, 1.0):
        return rf"$10^{{{exp}}}$"
    coeff_txt = f"{coeff:.1f}".rstrip("0").rstrip(".")
    return rf"${coeff_txt}\times 10^{{{exp}}}$"


def save_figure_stacked(base_name):
    pdf_path = OUTPUT_DIR / f"{base_name}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ Saved: {pdf_path.name}")


configure_plot_fonts(use_tex=True)

# Load data
results_path = Path("experiment_outputs_full/ac5_oos_ablation_results_full.csv")
summary_path = Path("experiment_outputs_full/ac5_oos_ablation_summary_full.csv")
best_sinkhorn_path = Path("experiment_outputs_full/ac5_oos_best_sinkhorn_vs_baselines_full.csv")

results_df = pd.read_csv(results_path)
summary_df = pd.read_csv(summary_path)
best_sinkhorn_df = pd.read_csv(best_sinkhorn_path)

# Extract unique values
rho_vals = sorted(results_df["rho_train"].unique())
eps_vals = sorted(summary_df["eps_ablation"].unique())
controllers = results_df["controller"].unique()

print(f"Found {len(rho_vals)} rho values: {rho_vals}")
print(f"Found {len(eps_vals)} epsilon values: {eps_vals}")
print(f"Found {len(controllers)} controllers: {controllers}")


# ============================================================================
# Cost and Violation Improvement (%)
# ============================================================================
fig5, axes5 = plt.subplots(1, 2, figsize=(17.5, 6.5))
# fig5.suptitle("Sinkhorn Improvement over Baselines (%)", fontsize=13, fontweight='bold')

ax = axes5[0]
cost_improve_vs_w = (best_sinkhorn_df["wasserstein_cost_minus_sinkhorn"].values / 
                     best_sinkhorn_df["wasserstein_mean_cost"].values * 100)
cost_improve_vs_n = (best_sinkhorn_df["nominal_cost_minus_sinkhorn"].values / 
                     best_sinkhorn_df["nominal_mean_cost"].values * 100)

x = np.arange(len(rho_vals))
width = 0.35
rho_tick_labels = [str(int(round(r * 1e3))) for r in rho_vals]

ax.bar(x - width/2, cost_improve_vs_w, width, label='Wasserstein', color='darkgreen', alpha=0.7)
ax.bar(x + width/2, cost_improve_vs_n, width, label='Empirical', color='red', alpha=0.7)

ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"Cost Increment (\%)")
ax.set_xticks(x)
ax.set_xticklabels(rho_tick_labels)
ax.text(0.98, -0.03, r"$\times 10^{-3}$", transform=ax.transAxes, ha="left", va="top")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

# Violation improvement (note: negative is better, showing reduction)
ax = axes5[1]
viol_improve_vs_w = (best_sinkhorn_df["wasserstein_violation_minus_sinkhorn"].values * 100)
viol_improve_vs_n = (best_sinkhorn_df["nominal_violation_minus_sinkhorn"].values * 100)

ax.bar(x - width/2, viol_improve_vs_w, width, label='Wasserstein', color='darkgreen', alpha=0.7)
ax.bar(x + width/2, viol_improve_vs_n, width, label='Empirical', color='red', alpha=0.7)

ax.set_xlabel(r"$\rho$")
ax.set_ylabel("Violation Increment (\%)")
ax.set_xticks(x)
ax.set_xticklabels(rho_tick_labels)
ax.text(0.98, -0.03, r"$\times 10^{-3}$", transform=ax.transAxes, ha="left", va="top")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

plt.tight_layout()
save_figure_stacked("05_sinkhorn_improvement")
plt.close()

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE: Best Sinkhorn vs Baselines")
print("="*80)
print(best_sinkhorn_df.to_string(index=False))
print("="*80)

print("\n✓ All visualizations completed successfully!")
print("Output location: experiment_outputs_full/")
