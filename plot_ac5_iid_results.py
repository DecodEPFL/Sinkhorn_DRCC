"""
plot_ac5_iid_results.py

Two publication-quality figures for the i.i.d. out-of-sample test.

Figure 1  (inspired by plot_ac5_oos_results.py)
    Best Sinkhorn vs Wasserstein vs Nominal:
    cost increment and violation-rate increment, one group per rho.

Figure 2  (inspired by plot_ac5_trajectories.py)
    Terminal-state pairwise scatter  (beta vs phi,  p vs r)
    for Sinkhorn / Wasserstein / Empirical on i.i.d. noise.

Prerequisites
-------------
1. Run ac5_iid_oos_test.m          → experiment_outputs_iid/iid_oos_results_*.csv
2. Run ac5_iid_build_plot_data.m   → experiment_outputs_iid/ac5_iid_plot_data.mat
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Paths  (auto-discover latest CSV; .mat path is fixed by MATLAB script)
# ---------------------------------------------------------------------------
IID_DIR   = Path("experiment_outputs_iid")
PLOT_MAT  = IID_DIR / "ac5_iid_plot_data.mat"
OUT_DIR   = IID_DIR

def _latest_csv(pattern):
    files = sorted(IID_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {IID_DIR / pattern}")
    return files[-1]

RESULTS_CSV = _latest_csv("iid_oos_results_*.csv")
print(f"Results CSV : {RESULTS_CSV}")
print(f"Plot MAT    : {PLOT_MAT}")

# ---------------------------------------------------------------------------
# Shared style  (mirrors both existing plotters)
# ---------------------------------------------------------------------------
def configure_plot_fonts(use_tex=True):
    has_latex = shutil.which("latex") is not None
    if use_tex and has_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif":  ["Times New Roman", "Times"],
            "axes.unicode_minus": False,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif":  ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
        })

configure_plot_fonts(use_tex=True)
plt.rcParams.update({
    "font.size":        18,
    "axes.labelsize":   19,
    "xtick.labelsize":  17,
    "ytick.labelsize":  17,
    "legend.fontsize":  17,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

def save_figure(fig, base_name):
    path = OUT_DIR / f"{base_name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Load & tidy results
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV)

def _ctrl_type(name):
    if name.startswith("Sinkhorn"):
        return "Sinkhorn"
    return name   # "Wasserstein" or "Nominal"

df["ctrl_type"] = df["controller"].apply(_ctrl_type)
rho_vals = sorted(df["rho_train"].unique())

# Best Sinkhorn per rho: lowest violation_rate, tie-break on mean_cost
def _best_sinkhorn(sub):
    s = sub[sub["ctrl_type"] == "Sinkhorn"].sort_values(
        ["violation_rate", "mean_cost"]
    )
    return s.iloc[0] if not s.empty else None

summary_rows = []
for rho in rho_vals:
    sub  = df[df["rho_train"] == rho]
    best = _best_sinkhorn(sub)
    wass = sub[sub["ctrl_type"] == "Wasserstein"]
    nom  = sub[sub["ctrl_type"] == "Nominal"]
    if best is None or wass.empty or nom.empty:
        continue
    summary_rows.append({
        "rho":              rho,
        "sinkhorn_cost":    best["mean_cost"],
        "sinkhorn_viol":    best["violation_rate"],
        "wasserstein_cost": wass["mean_cost"].values[0],
        "wasserstein_viol": wass["violation_rate"].values[0],
        "nominal_cost":     nom["mean_cost"].values[0],
        "nominal_viol":     nom["violation_rate"].values[0],
    })

cmp = pd.DataFrame(summary_rows)
print("\nSummary:")
print(cmp.to_string(index=False))


# ===========================================================================
# Figure 1 – cost increment & violation-rate increment
# (mirrors the bar style of plot_ac5_oos_results.py  Figure 5)
# ===========================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(17.5, 6.5))

x     = np.arange(len(cmp))
width = 0.35
rho_tick_labels = [str(int(round(r * 1e3))) for r in cmp["rho"].values]

# ── Panel A: cost increment (%) relative to Sinkhorn ─────────────────────
ax = axes1[0]
cost_w = (cmp["wasserstein_cost"] - cmp["sinkhorn_cost"]) / cmp["wasserstein_cost"] * 100
cost_n = (cmp["nominal_cost"]     - cmp["sinkhorn_cost"]) / cmp["nominal_cost"]     * 100

ax.bar(x - width/2, cost_w.values, width,
       label="Wasserstein", color="darkgreen", alpha=0.7)
ax.bar(x + width/2, cost_n.values, width,
       label="Empirical",   color="red",       alpha=0.7)

ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"Cost Increment (\%)")
ax.set_xticks(x)
ax.set_xticklabels(rho_tick_labels)
ax.text(0.98, -0.03, r"$\times 10^{-3}$", transform=ax.transAxes, ha="left", va="top")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# ── Panel B: violation-rate increment (pp) relative to Sinkhorn ──────────
ax = axes1[1]
viol_w = (cmp["wasserstein_viol"] - cmp["sinkhorn_viol"]) * 100
viol_n = (cmp["nominal_viol"]     - cmp["sinkhorn_viol"]) * 100

ax.bar(x - width/2, viol_w.values, width,
       label="Wasserstein", color="darkgreen", alpha=0.7)
ax.bar(x + width/2, viol_n.values, width,
       label="Empirical",   color="red",       alpha=0.7)

ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"Violation Increment (\%)")
ax.set_xticks(x)
ax.set_xticklabels(rho_tick_labels)
ax.text(0.98, -0.03, r"$\times 10^{-3}$", transform=ax.transAxes, ha="left", va="top")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save_figure(fig1, "iid_fig1_sinkhorn_improvement")
plt.close(fig1)


# ===========================================================================
# Figure 2 – terminal-state pairwise scatter
# (mirrors plot_ac5_trajectories.py  plot_terminal_pairwise)
# ===========================================================================
if not PLOT_MAT.is_file():
    print(f"\nSkipping Figure 2: {PLOT_MAT} not found.")
    print("Run ac5_iid_build_plot_data.m first, then rerun this script.")
else:
    data         = loadmat(str(PLOT_MAT), squeeze_me=False)
    trajectories = data["trajectories"]   # [n_ctrl, n_samples, N, d]
    state_ub     = np.ravel(data["state_ub"]).astype(float)

    # Controller names from MATLAB cell array
    raw_names = data["controller_names"]
    names = []
    for item in np.ravel(raw_names):
        if isinstance(item, np.ndarray):
            names.append("".join(item.tolist()).strip())
        else:
            names.append(str(item).strip())

    n_ctrl   = trajectories.shape[0]
    term     = trajectories[:, :, -1, :]        # [n_ctrl, n_samples, d]
    term_deg = np.degrees(term)                 # convert once; reused per panel/controller
    ub_deg   = np.degrees(state_ub)             # convert once; reused per panel

    pairs  = [(0, 3), (1, 2)]            # (beta, phi) and (p, r)
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]

    STATE_SYMBOLS = [r"\beta", r"p", r"r", r"\phi"]

    def _label(idx):
        sym = STATE_SYMBOLS[idx] if idx < len(STATE_SYMBOLS) else f"x_{{{idx+1}}}"
        return rf"${sym}$"

    fig2 = plt.figure(figsize=(13, 5.5))
    axs  = [
        fig2.add_axes([0.08, 0.16, 0.40, 0.70]),
        fig2.add_axes([0.56, 0.16, 0.40, 0.70]),
    ]

    for k, (i, j) in enumerate(pairs):
        ax = axs[k]

        ub_i_deg = ub_deg[i]
        ub_j_deg = ub_deg[j]

        ax.fill([-ub_i_deg, ub_i_deg, ub_i_deg, -ub_i_deg],
                [-ub_j_deg, -ub_j_deg, ub_j_deg, ub_j_deg],
                color="0.85", alpha=0.35, zorder=0)

        for c in range(n_ctrl):
            ax.scatter(term_deg[c, :, i], term_deg[c, :, j], s=16, alpha=0.75,
                       color=colors[c % len(colors)], label=names[c])

        ax.plot([-ub_i_deg, ub_i_deg,  ub_i_deg, -ub_i_deg, -ub_i_deg],
                [-ub_j_deg, -ub_j_deg, ub_j_deg,  ub_j_deg, -ub_j_deg],
                "k--", lw=1)

        ax.set_xlabel(_label(i))
        ax.set_ylabel(_label(j))
        ax.grid(alpha=0.25)

        pad_i = 0.2 * ub_i_deg
        pad_j = 0.2 * ub_j_deg
        ax.set_xlim(-ub_i_deg - pad_i, ub_i_deg + pad_i)
        ax.set_ylim(-ub_j_deg - pad_j, ub_j_deg + pad_j)

    handles, labels_leg = axs[0].get_legend_handles_labels()
    if handles:
        fig2.legend(
            handles, labels_leg,
            loc="upper center", ncol=max(1, len(labels_leg)),
            bbox_to_anchor=(0.5, 0.98),
            frameon=True, fancybox=False,
            framealpha=1.0, facecolor="white", edgecolor="black",
        )

    save_figure(fig2, "iid_fig2_terminal_pairwise")
    plt.close(fig2)

print("\nDone.")
