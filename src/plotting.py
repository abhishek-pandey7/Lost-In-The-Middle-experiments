"""Plotting utilities for position-bias curves."""
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_curve(
    x_values,
    y_values,
    title: str,
    save_path: str,
    xlabel: str = "Position (0=start, 1=end)",
    ylabel: str = "Accuracy",
    ylim: tuple = (-0.05, 1.05),
    color: str = "#E63946",
):
    """Plot a standard position-bias accuracy curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2.5, markersize=10, color=color)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=13)
    plt.ylim(ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Plot saved: {save_path}")


def plot_bar(categories, values, title: str, save_path: str, ylabel: str = "Accuracy", ylim=(0, 1.05), colors=None):
    """Plot a bar chart (e.g., for multi-needle start/middle/end)."""
    if colors is None:
        colors = ["#2E86AB", "#E63946", "#2E86AB"]
    plt.figure(figsize=(6, 5))
    plt.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=13)
    plt.ylim(ylim)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Bar plot saved: {save_path}")


def plot_multi_curves(curves, labels, title, save_path, xlabel="Position", ylabel="Accuracy"):
    """Overlay multiple curves for comparison."""
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, (x, y, label) in enumerate(zip(curves["x"], curves["y"], labels)):
        plt.plot(x, y, marker="o", linewidth=2.0, markersize=8, label=label, color=cmap(i))
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=13)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Multi-curve plot saved: {save_path}")
