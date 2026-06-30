"""figstyle.py -- shared publication style for ACS Nano-quality figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# ---- global rcParams ----
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#2b2b2b",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.6,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,        # editable text in vector output
    "ps.fonttype": 42,
})

# ---- brand palette ----
INK = "#1d2433"
SUBINK = "#5b6472"
GRID = "#e6e9ef"

# per-channel identity colours (vivid but tasteful)
CH = {
    "7AHL": "#0E9594",   # teal
    "3EMN": "#E07A1F",   # amber  (VDAC)
    "2ZW3": "#8E44AD",   # violet (connexin)
    "MspA": "#2E86DE",   # blue
}
CH_LIGHT = {k: to_rgba(v, 0.18) for k, v in CH.items()}

ACCENT = "#E84855"       # red accent
GOLD = "#F4C20D"
GREEN = "#16A085"

# sequential map for the design-map heatmap (white-teal-indigo-magenta)
DESIGN_CMAP = LinearSegmentedColormap.from_list(
    "design", ["#f7fbfd", "#bde7e3", "#3fb8af", "#2e6fb0",
               "#5b3a9e", "#9b2d8f", "#d81b60"])

# warm shape-render colormap
SHAPE_CMAP = LinearSegmentedColormap.from_list(
    "shape", ["#3a1c71", "#7b2ff7", "#d76d77", "#ffaf7b", "#ffd86b"])


def style_ax(ax, grid=True):
    ax.tick_params(colors=INK, length=3)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
    if grid:
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    return ax


def panel_label(ax, letter, dx=-0.16, dy=1.06, size=12):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left", color=INK)


def shade_surface(ax, X, Y, Z, base_color, light=(0.3, 0.5, 0.8), alpha=1.0,
                  lw=0.0):
    """Lambertian-shaded 3D surface for a polished solid look."""
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=None, blend_mode="soft") if False else None
    base = np.array(to_rgba(base_color)[:3])
    # compute simple normals via gradient for shading
    nz = np.ones_like(Z)
    fc = np.empty(Z.shape + (4,))
    # per-face shading using surface normal . light
    ln = np.array(light) / np.linalg.norm(light)
    gx, gy = np.gradient(Z)
    nx, ny = -gx, -gy
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    shade = (nx * ln[0] + ny * ln[1] + nz * ln[2]) / norm
    shade = 0.55 + 0.45 * np.clip(shade, 0, 1)
    for i in range(3):
        fc[..., i] = np.clip(base[i] * shade, 0, 1)
    fc[..., 3] = alpha
    surf = ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                           linewidth=lw, antialiased=True, shade=False)
    return surf


def superquadric_mesh(rx, ry, c, e1, e2, nu=60, nv=120):
    u = np.linspace(-np.pi / 2, np.pi / 2, nu)
    v = np.linspace(-np.pi, np.pi, nv)
    U, V = np.meshgrid(u, v)

    def f(ang, e):
        return (np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** e,
                np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** e)
    cU, sU = f(U, e1)
    cV, sV = f(V, e2)
    X = rx * cU * cV
    Y = ry * cU * sV
    Z = c * sU
    return X, Y, Z


def set_equal_3d(ax, X, Y, Z):
    xmid = (X.max() + X.min()) / 2; ymid = (Y.max() + Y.min()) / 2
    zmid = (Z.max() + Z.min()) / 2
    r = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()) / 2 * 1.05
    ax.set_xlim(xmid - r, xmid + r); ax.set_ylim(ymid - r, ymid + r)
    ax.set_zlim(zmid - r, zmid + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()
