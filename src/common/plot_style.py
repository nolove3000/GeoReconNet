"""Shared publication plotting style for all formal model routes."""

from pathlib import Path

from cycler import cycler
import matplotlib
from matplotlib import font_manager


BLUE_PALETTE = (
    "#0B3C5D",
    "#2F75B5",
    "#5B9BD5",
    "#7FB3D5",
    "#A9CCE3",
)

PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.grid": False,
    "axes.prop_cycle": cycler(color=BLUE_PALETTE),
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "savefig.bbox": "tight",
}


def register_times_new_roman():
    """Register Microsoft core fonts when Matplotlib's cache has not discovered them."""
    font_directory = Path("/usr/share/fonts/truetype/ms-core-fonts")
    for font_path in sorted(font_directory.glob("Times*.TTF")):
        font_manager.fontManager.addfont(font_path)


def configure_plot_style():
    """Apply the formal Times New Roman, blue-tone, grid-free style."""
    register_times_new_roman()
    matplotlib.rcParams.update(PLOT_STYLE)


def remove_axes_grid(axes):
    """Explicitly disable major and minor grids on one or more axes."""
    try:
        axes_iterator = axes.flat
    except AttributeError:
        axes_iterator = (axes,)
    for axis in axes_iterator:
        axis.grid(False, which="both")
