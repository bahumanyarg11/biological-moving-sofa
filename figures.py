"""
figures.py -- full research-grade figure suite (Figs 1-9 + TOC).
600-dpi PNG + vector PDF. Run after the analysis pipeline has written results/.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyArrowPatch, Circle, Ellipse, Rectangle,
                                FancyBboxPatch, Polygon, PathPatch)
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
import figstyle as F
import geom

OUT = "figs/"
os.makedirs(OUT, exist_ok=True)
CH = ["7AHL", "3EMN", "2ZW3", "MspA"]
NAME = {"7AHL": r"$\alpha$-Hemolysin", "3EMN": "VDAC1",
        "2ZW3": "Connexin-26", "MspA": "MspA"}
FILES = {"7AHL": "7ahl.cif", "3EMN": "3emn.cif",
         "2ZW3": "2zw3.cif", "MspA": "1uun_bio.cif"}

geo = json.load(open("results/channel_geometry.json"))
opt = json.load(open("results/optimization.json"))
val = json.load(open("results/validation.json"))
dmap = json.load(open("results/designmap.json"))
meso = json.load(open("results/mesoscale.json"))
bd = json.load(open("results/bd.json")) if os.path.exists("results/bd.json") else None
WHITE = [pe.withStroke(linewidth=2.2, foreground="white")]


def cross_section_2d(ax, pid, color, slab=3.5, show_rod=True):
    """Clean 2D longitudinal cross-section: protein atoms (silhouette), lumen
    band, and the optimal nanorod threading the pore. Axial = vertical."""
    from matplotlib.collections import PatchCollection
    g = geo[pid]
    coords, radii = geom.load_atoms("data/" + FILES[pid])
    axis = np.array(g["axis"]); axis /= np.linalg.norm(axis)
    u = np.array(g["u"]); v = np.array(g["v"]); cmid = np.array(g["centerline_A"]).mean(0)
    rel = coords - cmid
    s = rel @ axis; a = rel @ u; b = rel @ v
    half = g["length_nm"] * 10 / 2
    m = (np.abs(b) < slab) & (np.abs(s) < half + 8) & (np.abs(a) < 26)
    circs = [plt.Circle((a[i]/10, s[i]/10), radii[i]/10) for i in np.where(m)[0]]
    pcat = PatchCollection(circs, facecolor="#c7ccd6", edgecolor="white",
                           lw=0.12, alpha=0.95, zorder=2)
    ax.add_collection(pcat)
    # lumen band
    cl = np.array(g["centerline_A"]) - cmid
    sl = cl @ axis; al = cl @ u; R = np.array(g["R_A"])
    order = np.argsort(sl); sl, al, R = sl[order], al[order], R[order]
    ax.fill_betweenx(sl/10, (al-R)/10, (al+R)/10, color=color, alpha=0.30,
                     zorder=3, lw=0)
    ax.plot((al-R)/10, sl/10, color=color, lw=1.0, zorder=4)
    ax.plot((al+R)/10, sl/10, color=color, lw=1.0, zorder=4)
    # rod
    if show_rod:
        o = opt[pid]["opt_at_length"]
        rx = o["rx"]; c = min(val[pid]["a_nav_nm"], g["length_nm"]/2*1.05)
        bc = (np.array(g["bottleneck_center"]) - cmid)
        ca = bc @ u / 10
        rod = FancyBboxPatch((ca - rx, -c), 2*rx, 2*c,
                             boxstyle=f"round,pad=0,rounding_size={rx*0.9:.3f}",
                             fc=F.ACCENT, ec="white", lw=1.0, zorder=6, alpha=0.96)
        ax.add_patch(rod)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-half/10-0.8, half/10+0.8)


def save(fig, name):
    fig.savefig(OUT + name + ".png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


# ============================================================ Fig 1: concept
def fig1():
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22, left=0.04,
                          right=0.98, bottom=0.05, top=0.92)

    # (a) moving sofa
    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.1, 2.25); ax.set_ylim(-0.1, 2.25)
    ax.add_patch(Rectangle((0, 0), 2.15, 1.0, fc="#eaf6f5", ec="none"))
    ax.add_patch(Rectangle((0, 0), 1.0, 2.15, fc="#eaf6f5", ec="none"))
    for seg in [([0, 2.15], [1, 1]), ([1, 1], [1, 2.15]),
                ([0, 0], [0, 2.15]), ([0, 2.15], [0, 0])]:
        ax.plot(seg[0], seg[1], color=F.SUBINK, lw=1.1)
    th = np.linspace(0, np.pi, 120)
    ax.fill(0.5 + 0.62*np.cos(th), 0.5 + 0.62*np.sin(th), color=F.ACCENT,
            alpha=0.92, zorder=4)
    ax.add_patch(Circle((0.5, 0.5), 0.30, fc="#eaf6f5", ec="none", zorder=5))
    ax.add_patch(FancyArrowPatch((1.5, 0.5), (1.02, 1.0), color=F.INK,
                 arrowstyle="-|>", mutation_scale=12, lw=1.5, zorder=6))
    ax.text(1.1, 2.05, "Moser's moving sofa\n(largest area around a corner)",
            fontsize=7.6, color=F.INK, ha="left", va="top", fontweight="bold")
    F.panel_label(ax, "a", dx=0.0, dy=1.0)

    # (b) sphere/rod/disc through aperture + volume bars
    ax = fig.add_subplot(gs[0, 1]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.5, 1.9); ax.set_ylim(-1.3, 1.45)
    Rap = 0.32
    ax.add_patch(Rectangle((-0.14, -1.25), 0.28, 2.5, fc="#cfd6e2",
                 ec=F.SUBINK, lw=0.8))
    ax.add_patch(Circle((0, 0), Rap, fc="white", ec=F.SUBINK, lw=0.8, zorder=3))
    ax.add_patch(Circle((-1.05, 0.8), Rap, fc=F.CH["MspA"], ec="white", lw=1))
    ax.text(-1.05, 0.8, "1x", color="white", ha="center", va="center",
            fontsize=7, fontweight="bold")
    ax.add_patch(FancyBboxPatch((-1.32, -0.12), 1.05, 0.24,
                 boxstyle="round,pad=0,rounding_size=0.12", fc=F.ACCENT, ec="white"))
    ax.text(-0.78, 0, "rod", color="white", ha="center", va="center",
            fontsize=7, fontweight="bold")
    ax.add_patch(Ellipse((-1.05, -0.82), 0.18, 0.62, fc=F.GREEN, ec="white"))
    ax.text(-0.7, -0.82, "disc", color=F.GREEN, ha="left", va="center",
            fontsize=7, fontweight="bold")
    ax.add_patch(FancyArrowPatch((0.18, 0), (0.55, 0), color=F.INK,
                 arrowstyle="-|>", mutation_scale=12, lw=1.6))
    # volume bars
    for i, (lab, v, c) in enumerate([("sphere", 1, F.CH["MspA"]),
                                     ("rod", 9, F.ACCENT), ("disc", 4, F.GREEN)]):
        y = 0.95 - i*0.62
        ax.add_patch(Rectangle((0.75, y-0.12), v/9*1.0, 0.24, fc=c, ec="none"))
        ax.text(0.75, y+0.2, f"{lab}", fontsize=6.4, color=F.INK)
    ax.text(0.2, 1.36, r"same aperture $\rightarrow$ more cargo volume",
            fontsize=7.6, color=F.INK, ha="center", va="top", fontweight="bold")
    F.panel_label(ax, "b", dx=0.0, dy=1.0)

    # (c) two constraints schematic
    ax = fig.add_subplot(gs[1, 0]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.2, 3.4); ax.set_ylim(-0.4, 2.3)
    t = np.linspace(0, 1, 200); cx = 0.3 + 2.8*t
    cy = 1.0 + 0.5*np.sin(2.1*t*np.pi)*(0.3+0.7*t)
    dxc = np.gradient(cx); dyc = np.gradient(cy); nl = np.hypot(dxc, dyc)
    nx = -dyc/nl; ny = dxc/nl; w = 0.27
    top = np.column_stack([cx+w*nx, cy+w*ny]); bot = np.column_stack([cx-w*nx, cy-w*ny])[::-1]
    ax.fill(*np.vstack([top, bot]).T, color=F.CH["2ZW3"], alpha=0.16)
    ax.plot(cx, cy, color=F.SUBINK, lw=0.8, ls=(0, (4, 3)))
    i0 = 92; seg = 40
    ax.plot([cx[i0-seg], cx[i0+seg]], [cy[i0-seg], cy[i0+seg]], color=F.ACCENT,
            lw=8, solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(cx[i0+seg], cy[i0+seg]+0.34), xytext=(cx[i0-seg], cy[i0-seg]+0.34),
                arrowprops=dict(arrowstyle="<->", color=F.INK, lw=1.0))
    ax.text(cx[i0], cy[i0]+0.5, "navigable length $2a$\n(moving-sofa limit)",
            fontsize=7, color=F.INK, ha="center")
    ax.annotate("", xy=(cx[6], cy[6]+w), xytext=(cx[6], cy[6]-w),
                arrowprops=dict(arrowstyle="<->", color=F.INK, lw=1.0))
    ax.text(cx[6]-0.06, cy[6]+0.44, r"$R_{\rm bot}$", fontsize=7.5, ha="right")
    ax.text(1.7, 2.25, "rigid particle in a real channel", fontsize=7.6,
            color=F.INK, ha="center", va="top", fontweight="bold")
    F.panel_label(ax, "c", dx=0.0, dy=1.0)

    # (d) scale ladder
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rungs = [("1 nm", "porins / nanopores", F.CH["7AHL"]),
             ("3 nm", "nuclear pore (passive)", F.CH["MspA"]),
             ("40 nm", "nuclear pore (large cargo)", F.CH["2ZW3"]),
             ("100 nm", "mucus / ECM mesh", F.GREEN),
             ("600 nm", "tumour vascular gaps", F.CH["3EMN"])]
    for i, (s, lab, c) in enumerate(rungs):
        y = 0.86 - i*0.18
        ax.add_patch(Circle((0.12, y), 0.035, fc=c, ec="white", lw=1))
        ax.text(0.2, y, f"{s}  —  {lab}", fontsize=7.2, va="center", color=F.INK)
    ax.plot([0.12, 0.12], [0.86-4*0.18, 0.86], color=F.SUBINK, lw=1.2, zorder=0)
    ax.text(0.16, 1.0, "one scale-invariant principle", fontsize=7.6,
            color=F.INK, va="top", fontweight="bold")
    F.panel_label(ax, "d", dx=-0.02, dy=1.02)
    save(fig, "Figure1_concept")


# ====================================================== Fig 2: pipeline
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 2.0)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 2)
    steps = [("Atomic\nstructure (PDB)", F.CH["MspA"]),
             ("Inscribed-sphere\nlumen profile", F.CH["7AHL"]),
             ("Superquadric\nmorphospace", F.SHAPE_CMAP(0.6)),
             ("Navigability\noptimisation", F.CH["2ZW3"]),
             ("Brownian-dynamics\ntranslocation", F.CH["3EMN"]),
             ("Atomistic\nvalidation", F.GREEN)]
    n = len(steps); w = 1.4; gap = (10 - n*w)/(n+1)
    for i, (lab, c) in enumerate(steps):
        x = gap + i*(w+gap)
        ax.add_patch(FancyBboxPatch((x, 0.55), w, 0.9,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=F.to_rgba(c, 0.16) if isinstance(c, str) else c,
                     ec=c if isinstance(c, str) else F.INK, lw=1.4))
        ax.text(x+w/2, 1.0, lab, ha="center", va="center", fontsize=7,
                color=F.INK, fontweight="bold")
        if i < n-1:
            ax.add_patch(FancyArrowPatch((x+w, 1.0), (x+w+gap, 1.0),
                         arrowstyle="-|>", mutation_scale=12, lw=1.6,
                         color=F.SUBINK))
    ax.text(0.1, 1.9, "Computational pipeline", fontsize=9, fontweight="bold",
            color=F.INK, va="top")
    save(fig, "Figure2_pipeline")


# ============================================ Fig 3: molecular geometry
def fig3():
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0], hspace=0.3,
                          wspace=0.22, left=0.07, right=0.98, bottom=0.11, top=0.9)
    for j, pid in enumerate(CH):
        col = F.CH[pid]
        ax = fig.add_subplot(gs[0, j]); ax.axis("off")
        molf = f"figs/mol_{pid}.png"
        if os.path.exists(molf):
            ax.imshow(plt.imread(molf))
        else:
            cross_section_2d(ax, pid, col)
        ax.set_title(NAME[pid], color=col, fontsize=8.3, fontweight="bold", pad=2)
    fig.text(0.065, 0.9, "a", fontsize=12, fontweight="bold")
    fig.text(0.53, 0.945, "Molecular cutaways: optimal nanorod (red) threading the channel lumen",
             ha="center", fontsize=8.4, fontweight="bold", color=F.INK)
    for j, pid in enumerate(CH):
        g = geo[pid]; col = F.CH[pid]
        ax = fig.add_subplot(gs[1, j])
        z = np.array(g["z_nm"]); R = np.array(g["R_nm"])
        ax.fill_between(z, -R, R, color=col, alpha=0.22, lw=0)
        ax.plot(z, R, color=col, lw=1.5); ax.plot(z, -R, color=col, lw=1.5)
        ax.scatter([g["bottleneck_z_nm"]], [0], s=18, color=F.INK, zorder=5)
        ax.axhline(0, color=F.SUBINK, lw=0.5, ls=":")
        F.style_ax(ax); ax.set_ylim(-1.6, 1.6); ax.set_xlabel("axial z (nm)")
        ax.text(0.04, 0.96, f"$d$={2*g['bottleneck_R_nm']:.2f} nm\n$\\tau$={g['tortuosity']:.2f}",
                transform=ax.transAxes, fontsize=6.4, va="top", color=F.SUBINK)
        if j == 0:
            ax.set_ylabel("pore radius (nm)")
    fig.text(0.06, 0.45, "b", fontsize=12, fontweight="bold")
    save(fig, "Figure3_geometry")


# ============================================ Fig 4: morphospace + optima
def fig4():
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.05,
                          left=0.02, right=0.98, bottom=0.06, top=0.9)
    # (a) superquadric morphospace grid
    axL = fig.add_subplot(gs[0]); axL.axis("off")
    axL.set_xlim(0, 5); axL.set_ylim(0, 5)
    e1s = [0.2, 0.5, 1.0]; e2s = [0.2, 0.5, 1.0]
    sub = fig.add_gridspec(3, 3, left=0.04, right=0.5, bottom=0.1, top=0.82,
                           hspace=0.05, wspace=0.05)
    for a, e1 in enumerate(e1s):
        for b, e2 in enumerate(e2s):
            ax = fig.add_subplot(sub[a, b], projection="3d")
            X, Y, Z = F.superquadric_mesh(1, 1, 1.0, e1, e2, nu=34, nv=64)
            F.shade_surface(ax, X, Y, Z, F.SHAPE_CMAP(0.3+0.5*(a+b)/4))
            F.set_equal_3d(ax, X, Y, Z); ax.view_init(elev=20, azim=-60)
    fig.text(0.27, 0.88, "Superquadric morphospace", ha="center",
             fontsize=8.6, fontweight="bold", color=F.INK)
    fig.text(0.27, 0.05, r"$\epsilon_2$ (transverse squareness) $\rightarrow$",
             ha="center", fontsize=7.4, color=F.SUBINK)
    fig.text(0.025, 0.45, r"$\epsilon_1$ (axial) $\rightarrow$", va="center",
             rotation=90, fontsize=7.4, color=F.SUBINK)
    fig.text(0.02, 0.9, "a", fontsize=12, fontweight="bold")

    # (b) optimal nanorods vs sphere gallery
    sub2 = fig.add_gridspec(2, 2, left=0.54, right=0.97, bottom=0.08, top=0.82,
                            hspace=0.25, wspace=0.1)
    for k, pid in enumerate(CH):
        ax = fig.add_subplot(sub2[k//2, k % 2], projection="3d")
        o = opt[pid]["opt_at_length"]; col = F.CH[pid]
        rx = o["rx"]; c = val[pid]["a_nav_nm"]
        X, Y, Z = F.superquadric_mesh(rx, rx, c, o["e1"], o["e2"], nu=34, nv=66)
        F.shade_surface(ax, X, Z, Y, col)
        Rs = opt[pid]["sphere_R_nm"]
        sx, sy, sz = F.superquadric_mesh(Rs, Rs, Rs, 1, 1, nu=20, nv=40)
        F.shade_surface(ax, sx, sz, sy - 2.6*rx, "#aab2c0")
        allx = np.r_[X.ravel(), sx.ravel()]; allz = np.r_[Z.ravel(), sz.ravel()]
        ally = np.r_[Y.ravel(), (sy-2.6*rx).ravel()]
        F.set_equal_3d(ax, allx, allz, ally); ax.view_init(elev=20, azim=-64)
        ax.set_title(f"{NAME[pid]}  {val[pid]['gain_nav']:.0f}x", color=col,
                     fontsize=7.6, fontweight="bold", pad=-6)
    fig.text(0.75, 0.88, "Optimal shape = nanorod (sphere in grey)",
             ha="center", fontsize=8.2, fontweight="bold", color=F.INK)
    fig.text(0.52, 0.9, "b", fontsize=12, fontweight="bold")
    save(fig, "Figure4_morphospace")


# ============================================ Fig 5: capacity gains
def fig5():
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.07, right=0.98,
                          bottom=0.18, top=0.86)
    gains = [val[p]["gain_nav"] for p in CH]; cols = [F.CH[p] for p in CH]
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(range(4), gains, color=cols, width=0.66, edgecolor="white", lw=1.2)
    for b, gv, c in zip(bars, gains, cols):
        ax.text(b.get_x()+b.get_width()/2, gv+0.4, f"{gv:.1f}x", ha="center",
                fontsize=7.6, fontweight="bold", color=c)
    ax.axhline(1, color=F.SUBINK, lw=1, ls="--")
    F.style_ax(ax); ax.set_xticks(range(4))
    ax.set_xticklabels([NAME[p] for p in CH], rotation=18, fontsize=6.8, ha="right")
    ax.set_ylabel("capacity gain (x)"); ax.set_ylim(0, max(gains)*1.18)
    ax.set_title("Validated gain", fontsize=8.4, color=F.INK); F.panel_label(ax, "a")

    ax = fig.add_subplot(gs[1])
    for pid in CH:
        AR = np.array(opt[pid]["AR_grid"]); G = np.array(opt[pid]["gain_grid"])
        ax.plot(AR, G, color=F.CH[pid], lw=1.7)
        ax.scatter([val[pid]["AR_nav"]], [val[pid]["gain_nav"]], s=26,
                   color=F.CH[pid], edgecolor="white", lw=1, zorder=5)
    F.style_ax(ax); ax.set_xlabel(r"aspect ratio $a/R_{\rm bot}$")
    ax.set_ylabel("capacity gain (x)")
    ax.set_title("Gain $\\propto$ elongation", fontsize=8.4, color=F.INK)
    F.panel_label(ax, "b")

    ax = fig.add_subplot(gs[2])
    LR = [geo[p]["length_nm"]/(2*geo[p]["bottleneck_R_nm"]) for p in CH]
    loff = {"7AHL": (-6, 8), "3EMN": (6, 6), "2ZW3": (6, -3), "MspA": (6, -11)}
    lha = {"7AHL": "right", "3EMN": "left", "2ZW3": "left", "MspA": "left"}
    for p, lr, gv in zip(CH, LR, gains):
        ax.scatter([lr], [gv], s=80, color=F.CH[p], edgecolor="white", lw=1.2, zorder=4)
        ax.annotate(NAME[p], (lr, gv), loff[p], textcoords="offset points",
                    fontsize=6.4, color=F.CH[p], ha=lha[p])
    ax.set_xlim(min(LR)-1.2, max(LR)+1.6)
    xs = np.linspace(min(LR)*0.9, max(LR)*1.05, 50)
    ax.plot(xs, 1.5*xs, color=F.SUBINK, lw=1, ls="--", zorder=1)
    ax.text(0.95, 0.1, r"$\sim\!1.5\,L/2R_{\rm bot}$", transform=ax.transAxes,
            ha="right", fontsize=6.8, color=F.SUBINK)
    F.style_ax(ax); ax.set_xlabel(r"channel $L/2R_{\rm bot}$")
    ax.set_ylabel("capacity gain (x)")
    ax.set_title("Gain tracks channel geometry", fontsize=8.4, color=F.INK)
    F.panel_label(ax, "c")
    save(fig, "Figure5_gains")


# ============================================ Fig 6: navigability physics
def fig6():
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1, 1.15], wspace=0.36,
                          left=0.06, right=0.98, bottom=0.17, top=0.88)
    # (a) ladder in bent tube schematic
    ax = fig.add_subplot(gs[0]); ax.set_aspect("equal"); ax.axis("off")
    rho_in, rho_out = 1.0, 1.7
    th = np.linspace(0, np.pi/2, 80)
    for r, c in [(rho_in, F.SUBINK), (rho_out, F.SUBINK)]:
        ax.plot(r*np.cos(th), r*np.sin(th), color=c, lw=1.2)
    ax.fill(np.r_[rho_out*np.cos(th), rho_in*np.cos(th[::-1])],
            np.r_[rho_out*np.sin(th), rho_in*np.sin(th[::-1])],
            color=F.CH["3EMN"], alpha=0.13)
    # tangent chord (the rod)
    a = 0.5*(rho_in+rho_out)
    x0, y0 = rho_in*np.cos(np.pi/4), rho_in*np.sin(np.pi/4)
    half = np.sqrt(rho_out**2 - rho_in**2)
    dirn = np.array([-np.sin(np.pi/4), np.cos(np.pi/4)])
    p1 = np.array([x0, y0]) + half*dirn; p2 = np.array([x0, y0]) - half*dirn
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=F.ACCENT, lw=6,
            solid_capstyle="round")
    ax.text(0.55, 1.75, r"$a_{\max}=2\sqrt{\rho\,\Delta}$", fontsize=8,
            color=F.INK)
    ax.set_xlim(-0.1, 1.9); ax.set_ylim(-0.1, 1.95)
    ax.set_title("Ladder-in-a-bend", fontsize=8.2, color=F.INK)
    F.panel_label(ax, "a", dx=-0.02)

    # (b) a_nav vs tortuosity: analytic curve + empirical points
    ax = fig.add_subplot(gs[1])
    from optimize import ar_cap_curvature
    taus = np.linspace(1.01, 2.0, 60)
    R = 0.5; L = 8.0
    ana = [min(ar_cap_curvature(t, t*L, 0.5*R, R), (L/2)/R) for t in taus]
    ax.plot(taus, ana, color=F.INK, lw=1.6, label="idealised\nbend (l.b.)")
    for pid in CH:
        ax.scatter([geo[pid]["tortuosity"]], [val[pid]["AR_nav"]], s=60,
                   color=F.CH[pid], edgecolor="white", lw=1.2, zorder=5)
    F.style_ax(ax); ax.set_xlabel(r"tortuosity $\tau$")
    ax.set_ylabel(r"max aspect ratio $a/R_{\rm bot}$")
    ax.set_title("Tortuosity throttles elongation", fontsize=8.2, color=F.INK)
    ax.legend(fontsize=6.2, loc="upper right")
    ax.text(0.04, 0.05, "points: atomistic\n(real lumen)", transform=ax.transAxes,
            fontsize=5.8, color=F.SUBINK, va="bottom")
    F.panel_label(ax, "b")

    # (c) design map (aspect ratio x aperture anisotropy)
    ax = fig.add_subplot(gs[2])
    AR = np.array(dmap["AR"]); eps = np.array(dmap["eps_ap"])
    Z = np.array(dmap["gain"]); A, E = np.meshgrid(AR, eps)
    pc = ax.pcolormesh(A, E, Z, cmap=F.DESIGN_CMAP, shading="gouraud",
                       vmin=1, vmax=Z.max())
    cs = ax.contour(A, E, Z, levels=[3, 6, 10, 15, 20, 25], colors="white",
                    linewidths=0.6, alpha=0.75)
    ax.clabel(cs, fmt="%dx", fontsize=5.6)
    for pid in CH:
        ax.scatter([val[pid]["AR_nav"]], [geo[pid]["aniso_ratio"]], s=85,
                   color=F.CH[pid], edgecolor="white", lw=1.6, zorder=6,
                   label=NAME[pid])
    ax.legend(fontsize=5.6, loc="upper left", frameon=True, framealpha=0.85,
              handletextpad=0.2, borderpad=0.3)
    ax.set_xlabel(r"aspect ratio $a/R_{\rm bot}$")
    ax.set_ylabel(r"aperture anisotropy $\epsilon_{\rm ap}$")
    ax.set_title("Design map", fontsize=8.2, color=F.INK)
    cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.05); cb.set_label("gain (x)", fontsize=7)
    F.panel_label(ax, "c")
    save(fig, "Figure6_navigability")


# ============================================ Fig 7: Brownian dynamics
def fig7():
    if bd is None:
        print("   (bd.json missing, skip Fig7)"); return
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, left=0.08, right=0.97,
                          bottom=0.1, top=0.92)
    SC = {"sphere": "#aab2c0", "rod_opt": F.ACCENT, "disc": F.GREEN}
    SL = {"sphere": "sphere", "rod_opt": "optimal rod", "disc": "disc"}
    # (a) PMF for a representative channel
    ax = fig.add_subplot(gs[0, 0]); pid = "7AHL"
    z = np.array(bd[pid]["z_nm"])

    def sm(a, w=5):
        a = np.asarray(a); k = np.ones(w)/w
        return np.convolve(a, k, mode="same")
    for s in ["sphere", "rod_opt", "disc"]:
        ax.plot(z, sm(bd[pid][s]["G"]), color=SC[s], lw=1.8, label=SL[s])
    F.style_ax(ax); ax.set_xlabel("axial position (nm)")
    ax.set_ylabel(r"PMF $G(z)$ ($k_BT$)")
    ax.set_title(f"Entropic barrier ({NAME[pid]})", fontsize=8.2, color=F.INK)
    ax.legend(fontsize=6.8); F.panel_label(ax, "a")

    # (b) barrier height per channel/shape
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(4); wbar = 0.26
    for i, s in enumerate(["sphere", "rod_opt", "disc"]):
        vals = [bd[p][s]["Gbarrier"] for p in CH]
        ax.bar(x + (i-1)*wbar, vals, wbar, color=SC[s], label=SL[s],
               edgecolor="white", lw=0.6)
    F.style_ax(ax); ax.set_xticks(x); ax.set_xticklabels([NAME[p] for p in CH],
                  rotation=18, fontsize=6.6, ha="right")
    ax.set_ylabel(r"barrier $\Delta G^{\ddagger}$ ($k_BT$)")
    ax.set_title("Translocation barrier", fontsize=8.2, color=F.INK)
    ax.legend(fontsize=6.6); F.panel_label(ax, "b")

    # (c) P_trans vs drive (representative channel)
    ax = fig.add_subplot(gs[1, 0]); pid = "7AHL"
    drives = [0.0, 0.5, 1.0, 1.5, 2.0]
    for s in ["sphere", "rod_opt", "disc"]:
        P = [bd[pid][s]["Pf"][f"f{f}"]["P"] for f in drives]
        ax.plot(drives, P, "-o", color=SC[s], lw=1.6, ms=4, label=SL[s])
    F.style_ax(ax); ax.set_xlabel(r"drive $f$ ($k_BT$/nm)")
    ax.set_ylabel("translocation prob."); ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Driven translocation ({NAME[pid]})", fontsize=8.2, color=F.INK)
    ax.legend(fontsize=6.6); F.panel_label(ax, "c")

    # (d) capacity gain vs translocation barrier (viable shapes only)
    ax = fig.add_subplot(gs[1, 1])
    for pid in CH:
        sb = bd[pid]["sphere"]["Gbarrier"]; rb = bd[pid]["rod_opt"]["Gbarrier"]
        ax.scatter([sb], [1.0], marker="o", s=52, color=F.CH[pid],
                   edgecolor="white", lw=0.9, zorder=4)
        ax.scatter([rb], [val[pid]["gain_nav"]], marker="s", s=52,
                   color=F.CH[pid], edgecolor="white", lw=0.9, zorder=4)
        ax.annotate("", xy=(rb, val[pid]["gain_nav"]), xytext=(sb, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=F.CH[pid], lw=1.0,
                                    alpha=0.6))
    ax.axhline(1, color=F.SUBINK, lw=1, ls="--")
    F.style_ax(ax); ax.set_xlabel(r"entropic barrier $\Delta G^{\ddagger}$ ($k_BT$)")
    ax.set_ylabel("capacity gain (x)")
    ax.set_title("Rod: more cargo, no barrier penalty", fontsize=8.2, color=F.INK)
    ax.text(0.97, 0.06, r"circle: sphere $\rightarrow$ square: rod",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
            color=F.SUBINK)
    F.panel_label(ax, "d")
    save(fig, "Figure7_brownian")


# ============================================ Fig 8: mesoscale
def fig8():
    fig = plt.figure(figsize=(7.4, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.5,
                          left=0.08, right=0.99, bottom=0.16, top=0.88)
    # (a) scale invariance
    ax = fig.add_subplot(gs[0])
    inv = meso["scale_invariance"]; R = np.array(inv["R_nm"])
    styles = {"tau1.05_eps1.0": ("straight, circular", F.CH["7AHL"], "-"),
              "tau1.3_eps1.0": ("mild bend", F.CH["MspA"], "--"),
              "tau1.05_eps2.5": ("slit aperture", F.ACCENT, "-"),
              "tau1.8_eps1.0": ("tortuous", F.GOLD, "-.")}
    for k, (lab, c, ls) in styles.items():
        if k in inv:
            ax.plot(R, inv[k], ls, color=c, lw=2.0, label=lab)
    ax.set_xscale("log"); F.style_ax(ax)
    ax.set_xlabel("pore radius (nm)"); ax.set_ylabel("capacity gain (x)")
    ax.set_ylim(0, None)
    ax.set_title("Scale invariance of the principle", fontsize=8.2, color=F.INK)
    ax.legend(fontsize=6.2, loc="lower center", ncol=2)
    for xv, t in [(1, "porin"), (20, "NPC"), (100, "mucus"), (300, "EPR")]:
        ax.axvline(xv, color=F.GRID, lw=0.8, zorder=0)
        ax.text(xv, ax.get_ylim()[1]*0.97, t, rotation=90, fontsize=5.6,
                va="top", ha="right", color=F.SUBINK)
    F.panel_label(ax, "a")

    # (b) biological barriers gain
    ax = fig.add_subplot(gs[1])
    sys = meso["systems"]
    order = sorted(sys, key=lambda s: s["scale"])
    ys = np.arange(len(order))
    cmap = F.DESIGN_CMAP
    cols = [cmap(0.15 + 0.7*i/len(order)) for i in range(len(order))]
    ax.barh(ys, [s["gain"] for s in order], color=cols, edgecolor="white", lw=0.8)
    for i, s in enumerate(order):
        ax.text(s["gain"]+0.05, i, f"{s['gain']:.1f}x", va="center",
                fontsize=6.4, color=F.INK)
    ax.set_yticks(ys); ax.set_yticklabels(
        [f"{s['name']} ({s['scale']:.0f} nm)" for s in order], fontsize=6.2)
    ax.axvline(1, color=F.SUBINK, lw=1, ls="--")
    F.style_ax(ax); ax.set_xlabel("capacity gain (x)")
    ax.set_xlim(0, max(s["gain"] for s in order)*1.18)
    ax.set_title("Barriers to delivery", fontsize=8.2, color=F.INK)
    F.panel_label(ax, "b", dx=-0.62)
    save(fig, "Figure8_mesoscale")


# ============================================ Fig 9: validation
def fig9():
    import molsurface as M
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.0], wspace=0.3,
                          left=0.07, right=0.98, bottom=0.17, top=0.88)
    ax = fig.add_subplot(gs[0])
    for pid in CH:
        ax.plot(val[pid]["r_factors"], val[pid]["radius_clearance_nm"],
                color=F.CH[pid], lw=1.7, label=NAME[pid])
    ax.axhspan(-1, 0, color=F.ACCENT, alpha=0.08); ax.axhline(0, color=F.ACCENT, lw=1, ls="--")
    ax.axvline(1, color=F.SUBINK, lw=0.8, ls=":")
    F.style_ax(ax); ax.set_xlabel(r"probe radius / $R_{\rm bot}$")
    ax.set_ylabel("clearance (nm)"); ax.set_ylim(-0.6, 0.7)
    ax.set_title("Transverse limit", fontsize=8.2, color=F.INK)
    ax.legend(fontsize=6.2, loc="lower left"); F.panel_label(ax, "a")

    ax = fig.add_subplot(gs[1])
    for pid in CH:
        ax.plot(val[pid]["c_grid_nm"], val[pid]["length_clearance_nm"],
                color=F.CH[pid], lw=1.7)
        ax.scatter([val[pid]["a_nav_nm"]], [0], s=22, color=F.CH[pid],
                   edgecolor="white", lw=1, zorder=6)
    ax.axhspan(-1, 0, color=F.ACCENT, alpha=0.08); ax.axhline(0, color=F.ACCENT, lw=1, ls="--")
    F.style_ax(ax); ax.set_xlabel("rod half-length $a$ (nm)")
    ax.set_ylabel("clearance (nm)"); ax.set_ylim(-0.6, 0.7)
    ax.set_title("Navigable length", fontsize=8.2, color=F.INK); F.panel_label(ax, "b")

    ax = fig.add_subplot(gs[2])
    try:
        pid = "2ZW3"
        cross_section_2d(ax, pid, F.CH[pid])
        ax.set_title(f"Collision-free fit ({NAME[pid]})", fontsize=8.2, color=F.INK)
    except Exception as e:
        ax.text(0.5, 0.5, "skip", transform=ax.transAxes, ha="center")
        print("   fig9 render", e)
    ax.text(0.0, 1.0, "c", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=F.INK)
    save(fig, "Figure9_validation")


# ============================================ TOC
def toc():
    fig = plt.figure(figsize=(3.3, 1.85)); ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off"); ax.set_xlim(0, 3.3); ax.set_ylim(0, 1.85)
    t = np.linspace(0, 1, 200); cx = 0.2 + 2.0*t; cy = 0.95 + 0.16*np.sin(2*np.pi*t)
    dxc = np.gradient(cx); dyc = np.gradient(cy); nl = np.hypot(dxc, dyc)
    nx = -dyc/nl; ny = dxc/nl; w = 0.22
    top = np.column_stack([cx+w*nx, cy+w*ny]); bot = np.column_stack([cx-w*nx, cy-w*ny])[::-1]
    ax.fill(*np.vstack([top, bot]).T, color=F.CH["2ZW3"], alpha=0.18)
    ax.plot(top[:, 0], top[:, 1], color=F.CH["2ZW3"], lw=1.2)
    ax.plot(bot[::-1, 0], bot[::-1, 1], color=F.CH["2ZW3"], lw=1.2)
    ax.plot([cx[55], cx[150]], [cy[55], cy[150]], color=F.ACCENT, lw=9,
            solid_capstyle="round")
    ax.add_patch(Circle((2.55, 1.35), 0.15, fc="#aab2c0"))
    ax.add_patch(FancyArrowPatch((2.5, 1.12), (2.5, 0.66), color=F.INK,
                 arrowstyle="-|>", mutation_scale=11, lw=1.4))
    ax.add_patch(FancyBboxPatch((2.32, 0.36), 0.5, 0.16,
                 boxstyle="round,pad=0,rounding_size=0.08", fc=F.ACCENT, ec="none"))
    ax.text(2.86, 0.95, "up to\n19x cargo", fontsize=8.5, color=F.INK,
            fontweight="bold", va="center")
    ax.text(0.08, 1.74, "Moving-sofa shape optimisation\nfor channel transport",
            fontsize=8.2, color=F.INK, fontweight="bold", va="top")
    save(fig, "TOC")


if __name__ == "__main__":
    print("Rendering figures...")
    for fn in [fig1, fig2, fig4, fig5, fig6, fig8, fig9, fig3, fig7, toc]:
        try:
            fn()
        except Exception as e:
            import traceback; print("FAILED", fn.__name__, e); traceback.print_exc()
    print("done.")
