import argparse
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


STATE_SYMBOLS = [r"\beta", r"p", r"r", r"\phi"]


def _state_label(idx):
    if 0 <= idx < len(STATE_SYMBOLS):
        return rf"${STATE_SYMBOLS[idx]}$"
    return rf"$x_{{{idx+1}}}$"


def configure_latex_font():
    has_latex = shutil.which("latex") is not None
    if has_latex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times"],
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "mathtext.fontset": "cm",
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times"],
            }
        )
    # Set font sizes for publication quality
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelsize"] = 19
    plt.rcParams["xtick.labelsize"] = 17
    plt.rcParams["ytick.labelsize"] = 17
    plt.rcParams["legend.fontsize"] = 17


def _matlab_cellstr_to_list(cell):
    out = []
    flat = np.ravel(cell)
    for item in flat:
        if isinstance(item, np.ndarray):
            if item.dtype.kind in {"U", "S"}:
                out.append("".join(item.tolist()).strip())
            else:
                out.append(str(item.squeeze()))
        else:
            out.append(str(item).strip())
    return out


def _safe_names(raw_names, n_ctrl):
    names = _matlab_cellstr_to_list(raw_names)
    if len(names) != n_ctrl:
        return [f"Controller {i+1}" for i in range(n_ctrl)]
    names = ["Empirical" if n.strip().lower() == "nominal" else n for n in names]
    return names


def plot_terminal_pairwise(trajectories, names, state_ub, out_dir):
    pairs = [(0, 3), (1, 2)]
    n_ctrl, _, _, _ = trajectories.shape

    controller_colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]

    fig = plt.figure(figsize=(13, 5.5))
    axs = [
        fig.add_axes([0.08, 0.16, 0.40, 0.70]),
        fig.add_axes([0.56, 0.16, 0.40, 0.70]),
    ]

    term = trajectories[:, :, -1, :]

    for k, (i, j) in enumerate(pairs):
        ax = axs[k]

        x0, x1 = -state_ub[i], state_ub[i]
        y0, y1 = -state_ub[j], state_ub[j]
        ax.fill([x0, x1, x1, x0], [y0, y0, y1, y1], color="0.85", alpha=0.35, zorder=0)

        for c in range(n_ctrl):
            color = controller_colors[c % len(controller_colors)]
            ax.scatter(term[c, :, i], term[c, :, j], s=16, alpha=0.75, color=color, label=names[c])

        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=1)
        # ax.set_title(rf"Terminal states: {_state_label(i)} vs {_state_label(j)}")
        ax.set_xlabel(_state_label(i))
        ax.set_ylabel(_state_label(j))
        ax.grid(alpha=0.25)

        x_pad = 0.2 * max(abs(x0), abs(x1))
        y_pad = 0.2 * max(abs(y0), abs(y1))
        ax.set_xlim(x0 - x_pad, x1 + x_pad)
        ax.set_ylim(y0 - y_pad, y1 + y_pad)
        ax.set_aspect("auto")

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=max(1, len(labels)),
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
        )

    save_figure(fig, out_dir / "ac5_terminal_pairwise")
    plt.close(fig)


def save_figure(fig, base_path):
    # fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot AC5 trajectories from exported MATLAB data.")
    parser.add_argument("--input", type=str, default="ac5_plot_data.mat", help="Input .mat from export_ac5_plot_data")
    parser.add_argument("--out-dir", type=str, default="ac5_figures", help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input MAT file not found: {input_path}\n"
            "Generate it from MATLAB first by running ac5_build_plot_data.m, then rerun this script."
        )

    data = loadmat(str(input_path))
    trajectories = data["trajectories"]  # [n_ctrl, n_samples, N, d]
    state_ub = np.ravel(data["state_ub"]).astype(float)
    names = _safe_names(data["controller_names"], trajectories.shape[0])

    plt.style.use("seaborn-v0_8-whitegrid")
    configure_latex_font()

    plot_terminal_pairwise(trajectories, names, state_ub, out_dir)

    print(f"Saved figure in: {out_dir / 'ac5_terminal_pairwise.pdf'}")


if __name__ == "__main__":
    main()
