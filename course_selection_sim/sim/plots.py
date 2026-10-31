"""统一的作图样式（中文字体、配色、细线条、浅色网格）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 分类配色（固定顺序，已用色盲安全校验）：蓝、橙、青、黄、品红
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def setup():
    cjk = [f.name for f in font_manager.fontManager.ttflist
           if any(k in f.name for k in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans SC"))]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": (cjk[:1] or []) + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    })


def style_axis(ax, ygrid_only=True):
    if ygrid_only:
        ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
