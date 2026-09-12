"""Plot saved stock shadow forces without simulation or resampling."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs/isaac/stock-solver-force-shadow/rows.jsonl")
    parser.add_argument("--output", type=Path, default=root / "outputs/figures/stock_solver_force_shadow")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    time = np.array([r["time_s"] for r in rows])
    normal = np.array([r["normal_foot_fz_N"] for r in rows])
    shadow = np.array([r["shadow_foot_fz_N"] for r in rows])
    if not (np.isfinite(normal).all() and np.isfinite(shadow).all() and np.all(np.diff(time) > 0)):
        raise ValueError("Force traces must be finite with strictly increasing time")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, sharey=True, layout="constrained")
    for i, (ax, foot) in enumerate(zip(axes.flat, ("FR", "FL", "RR", "RL"))):
        ax.plot(time, normal[:, i], color="#3976a8", lw=.8, alpha=.9, label="Normal-contact Fz")
        ax.plot(time, shadow[:, i], color="#d06b27", lw=.8, alpha=.9, label="Reconstructed solver residual")
        ax.axhline(0, color="#999999", lw=.5)
        ax.set_title(foot, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, time[-1])
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Recorded simulation time (s)")
        ax.set_ylabel("World vertical force (N)")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Stock PhysX: full Go2 + ARX5 contact rollout\nOfficial UMI policy · diagnostic channel only", fontsize=13)
    fig.supxlabel("All 3,400 physics samples shown; first 20 ms are warmup. Residual is not an original Gym sensor measurement.", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
