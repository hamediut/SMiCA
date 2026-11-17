"""
Shared look of the result plot windows, so every plot has the same fonts and line widths
and every saved figure has the same size - consistent figures for a paper.

Importing this module applies PAPER_STYLE to matplotlib; every plot window imports it, so
no window sets font sizes of its own (an explicit fontsize= would override the style).

The sizes are chosen for a figure saved at PANEL_SIZE per panel and shrunk to 50% in the
paper - a one-panel 7 in figure to one 3.5 in journal column, a two-panel 14 in figure to a
7 in page width - so every printed figure has axis labels of 9 pt, tick labels of 8 pt and
legends of 7 pt, whatever size its window had on screen.
"""

import matplotlib

# Resolution of every saved plot. Affects PNG/JPEG; a PDF stays vector (only rasterised
# parts, if any, use it).
SAVE_DPI = 500

# Size of ONE panel (axes) in a saved figure, in inches. A figure with n panels side by
# side is saved at n times this width (see figure_size()).
PANEL_SIZE = (7.0, 5.0)

PAPER_STYLE = {
    'font.size': 16,                # any other text
    'axes.labelsize': 18,           # x, y and colorbar labels      -> 9 pt in print
    'xtick.labelsize': 16,          # tick labels                    -> 8 pt in print
    'ytick.labelsize': 16,
    'legend.fontsize': 14,          # legend entries                 -> 7 pt in print
    'axes.titlesize': 16,           # titles: on screen only, save_figure() leaves them out
    'axes.titleweight': 'bold',
    'lines.linewidth': 2.0,         # data lines                     -> 1 pt in print
    'axes.linewidth': 1.2,          # frame around the axes
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6.0,
    'ytick.major.size': 6.0,
    'xtick.minor.width': 0.9,
    'ytick.minor.width': 0.9,
    'xtick.minor.size': 3.5,
    'ytick.minor.size': 3.5,
    # keeps the larger labels inside the figure (on screen and saved), for every window
    'figure.constrained_layout.use': True,
}

matplotlib.rcParams.update(PAPER_STYLE)

# Per-function line style in plots that show several SMDs together. When the phase is one
# big connected cluster, C2 = S2 and the two curves lie exactly on top of each other: S2 is
# therefore drawn twice as wide and underneath every other curve (zorder below the default
# 2), so it shows as a border on both sides of the C2 line drawn on top of it.
CURVE_STYLES = {
    's2': {'linewidth': 4.0, 'zorder': 1.9},
}


def curve_style(name: str) -> dict:
    """Extra plot() keyword arguments for one SMD curve, e.g. ax.plot(r, y, **curve_style('s2')).
    Empty for functions without a special style, so they keep the shared defaults."""
    return dict(CURVE_STYLES.get(name, {}))


def mark_phase_fraction(ax, phi: float) -> None:
    """
    Black circle at (r = 0, phi) on a RAW P_n(r) plot. Every raw SMD (S2, C2, L, P3, P4, P6)
    starts at the phase fraction phi - at r = 0 all its points sit on the same pixel - so
    this highlights the porosity. It gets no legend entry. Not for the scaled f_n(r) curves,
    which always start at 1.

    Used only on plots of ONE 2D image or ONE 3D volume (polytope results, RES/REV S2, the S2
    window) - not in the slice/time evolution window, where every slice would add a marker.
    """
    ax.plot([0], [phi], linestyle='none', marker='o', markersize=8, color='black',
            zorder=3,        # on top of every curve, including the wide S2 line
            clip_on=False,   # never cut off by the edge of the axes
            label='_nolegend_')


def figure_size(n_panels: int = 1):
    """Saved size, in inches, of a figure with n_panels panels side by side."""
    return (PANEL_SIZE[0] * n_panels, PANEL_SIZE[1])


def r_axis_label(ndim: int) -> str:
    """
    x-axis label for curves plotted against the distance r: '$r$ (pixels)' when the curves
    were computed on 2D images, '$r$ (voxels)' when they were computed on 3D volumes.
    r is set in italic (math text), like the r inside the y labels, e.g. $P_n(r)$.

    ndim is the dimensionality of the data each curve was computed on - for a slice/time
    evolution that is one slice or time step (2 for a 2D time series, 3 for a 3D one), not
    the whole stack.
    """
    return r'$r$ (pixels)' if ndim == 2 else r'$r$ (voxels)'


def save_figure(fig, file_path: str, n_panels: int = 1) -> None:
    """
    Save a plot window's figure for a paper and leave the window as it was.

    The figure is saved:
      - at figure_size(n_panels), NOT at its on-screen size: the Qt canvas resizes the figure
        to fit its window, so without this the same plot would be saved larger or smaller
        (and its text would print smaller or larger) depending on how big the window was;
      - at SAVE_DPI;
      - without its titles, which are only there to tell the plots apart on screen.
    Titles and size are restored afterwards, also if saving fails.
    """
    on_screen_size = fig.get_size_inches().copy()
    titles = [ax.title for ax in fig.axes if ax.title.get_visible()]
    try:
        for title in titles:
            title.set_visible(False)
        # forward=False: change only the figure, never resize the window it is shown in
        fig.set_size_inches(*figure_size(n_panels), forward=False)
        fig.savefig(file_path, dpi=SAVE_DPI, bbox_inches='tight')
    finally:
        for title in titles:
            title.set_visible(True)
        fig.set_size_inches(*on_screen_size, forward=False)
        fig.canvas.draw_idle()  # repaint the window at its own size again
