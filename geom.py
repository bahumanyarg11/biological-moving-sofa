"""
geom.py -- Rigorous extraction of biological channel lumen geometry.

Re-implements a HOLE-style (Smart & Sansom, 1996) inscribed-sphere pore
profiler from atomic coordinates, plus quantification of:
  - pore-radius profile R(z) along a data-derived channel axis,
  - centerline tortuosity tau = arc-length / axial-span,
  - bottleneck cross-sectional anisotropy (min/max directional clearance).

All lengths are reported in nanometres.

The previous pipeline produced spurious zero-radius bottlenecks; this module
fixes that by (i) restricting the profile to the enclosed (protein-lined)
region and (ii) optimising the sphere centre per slice.
"""
import numpy as np
import warnings
from Bio.PDB import MMCIFParser
from scipy.optimize import minimize

warnings.simplefilter("ignore")

# Bondi van der Waals radii (Angstrom)
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
       "P": 1.80, "F": 1.47, "CL": 1.75, "NA": 2.27, "MG": 1.73,
       "ZN": 1.39, "CA": 2.31, "SE": 1.90}
DEFAULT_VDW = 1.70


def load_atoms(path, heavy_only=True):
    """Return (coords[N,3] Angstrom, radii[N] Angstrom) for one model."""
    s = MMCIFParser(QUIET=True).get_structure("x", path)
    model = next(iter(s))
    coords, radii = [], []
    for atom in model.get_atoms():
        el = (atom.element or atom.get_name()[0]).strip().upper()
        if heavy_only and el == "H":
            continue
        # skip waters
        res = atom.get_parent()
        if res.id[0] == "W":
            continue
        coords.append(atom.coord)
        radii.append(VDW.get(el, DEFAULT_VDW))
    return np.asarray(coords, float), np.asarray(radii, float)


def principal_axes(coords):
    c = coords.mean(0)
    X = coords - c
    cov = X.T @ X / len(X)
    w, v = np.linalg.eigh(cov)          # ascending eigenvalues
    return c, v[:, ::-1], w[::-1]       # columns = axes, descending spread


def _clearance(center, coords, radii):
    """Largest sphere radius centred at `center` not overlapping any atom (A)."""
    d = np.linalg.norm(coords - center, axis=1) - radii
    return d.min()


def clearance_many(centers, coords, radii, chunk=4000):
    """Vectorised clearance for K query centres -> array[K] (Angstrom)."""
    centers = np.atleast_2d(centers)
    out = np.empty(len(centers))
    for i in range(0, len(centers), chunk):
        c = centers[i:i + chunk]
        d = np.linalg.norm(c[:, None, :] - coords[None, :, :], axis=2) - radii[None, :]
        out[i:i + chunk] = d.min(axis=1)
    return out


def _neg_clearance(xy, z, axis_u, axis_v, axis_w, origin, coords, radii):
    center = origin + xy[0] * axis_u + xy[1] * axis_v + z * axis_w
    return -_clearance(center, coords, radii)


def _ortho(d):
    d = d / np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def _enclosed(center, u, v, coords, R, shell=4.5, need=6):
    """True if the transverse plane around `center` is walled by protein
    (atoms present in >=need of 8 angular sectors within R+shell)."""
    d = coords - center
    du, dv = d @ u, d @ v
    rp = np.sqrt(du**2 + dv**2)
    near = rp < (R + shell)
    if near.sum() < 8:
        return False
    ang = np.arctan2(dv[near], du[near])
    sect = ((ang + np.pi) / (2 * np.pi / 8)).astype(int) % 8
    return np.unique(sect).size >= need


def profile_along(direction, origin, coords, radii, n_slices=120,
                  n_grid=21, polish=True):
    """Inscribed-sphere radius profile along `direction` through `origin`.

    For each axial slice the in-plane sphere centre is found by a bounded,
    two-level vectorised grid search. The search is confined to the protein
    cross-section at that slice (centred on the local atom centroid), which
    keeps the inscribed sphere inside the lumen rather than escaping to bulk
    solvent where the clearance objective is unbounded.

    `polish` adds a finer second-level grid. Returns (z, centers, R, u, v)."""
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    u, v = _ortho(d)
    proj = (coords - origin) @ d
    pu = (coords - origin) @ u
    pv = (coords - origin) @ v
    zlo, zhi = np.percentile(proj, 0.5), np.percentile(proj, 99.5)
    zs = np.linspace(zlo, zhi, n_slices)
    dz = (zhi - zlo) / n_slices
    window = max(3.0, 2.5 * dz)

    def search(z, guess, rad, ng):
        """Best in-plane (u,v) sphere centre near `guess` within radius `rad`."""
        gx = np.linspace(-rad, rad, ng)
        GX, GY = np.meshgrid(gx, gx)
        local = np.column_stack([GX.ravel() + guess[0], GY.ravel() + guess[1]])
        cand = origin + np.outer(local[:, 0], u) + np.outer(local[:, 1], v) + z * d
        cl = clearance_many(cand, coords, radii)
        k = int(np.argmax(cl))
        xy, R = local[k].copy(), cl[k]
        if polish:
            step = gx[1] - gx[0]
            fx = np.linspace(-step, step, 7)
            FX, FY = np.meshgrid(fx, fx)
            l2 = np.column_stack([FX.ravel() + xy[0], FY.ravel() + xy[1]])
            c2 = origin + np.outer(l2[:, 0], u) + np.outer(l2[:, 1], v) + z * d
            cl2 = clearance_many(c2, coords, radii)
            k2 = int(np.argmax(cl2))
            if cl2[k2] > R:
                xy, R = l2[k2], cl2[k2]
        return xy, R

    centers = [None] * n_slices
    Rs = np.full(n_slices, -1.0)

    # anchor: at the central slice pick the highest-clearance point that is
    # actually walled by protein (an enclosed lumen point), not a surface pocket.
    i0 = n_slices // 2
    m = np.abs(proj - zs[i0]) < window
    cu, cv = (pu[m].mean(), pv[m].mean()) if m.any() else (0.0, 0.0)
    spread = np.percentile(np.sqrt((pu[m] - cu)**2 + (pv[m] - cv)**2), 90) \
        if m.sum() > 4 else 15.0
    rA = float(np.clip(spread, 5.0, 25.0))
    gx = np.linspace(-rA, rA, max(n_grid, 25))
    GX, GY = np.meshgrid(gx, gx)
    loc = np.column_stack([GX.ravel() + cu, GY.ravel() + cv])
    cand = origin + np.outer(loc[:, 0], u) + np.outer(loc[:, 1], v) + zs[i0] * d
    cl = clearance_many(cand, coords, radii)
    order = np.argsort(cl)[::-1]
    xy0, R0 = loc[order[0]], cl[order[0]]
    for j in order[:60]:
        c3 = origin + loc[j, 0] * u + loc[j, 1] * v + zs[i0] * d
        if cl[j] > 1.2 and _enclosed(c3, u, v, coords, cl[j]):
            xy0, R0 = loc[j], cl[j]
            break
    centers[i0] = xy0; Rs[i0] = R0

    # propagate outward with continuity (HOLE-style stepping). The step radius
    # is capped so the centre cannot jump across the wall into bulk solvent, and
    # propagation in a given direction stops once the lumen is no longer walled.
    step_cap = max(3.0, 1.5 * dz)
    for direction_i in (+1, -1):
        prev = xy0
        i = i0 + direction_i
        while 0 <= i < n_slices:
            xy, R = search(zs[i], prev, step_cap, 9)
            c3 = origin + xy[0] * u + xy[1] * v + zs[i] * d
            if not _enclosed(c3, u, v, coords, max(R, 0.0)):
                break
            centers[i] = c3
            Rs[i] = R
            prev = xy
            i += direction_i
    centers[i0] = origin + xy0[0] * u + xy0[1] * v + zs[i0] * d
    # fill any unreached slices with placeholders (trimmed out downstream)
    for i in range(n_slices):
        if centers[i] is None:
            centers[i] = origin + zs[i] * d
    return zs, np.array(centers), np.array(Rs), u, v


def _longest_interior_run(R, floor=1.2, mouth=18.0, gap=4):
    """Indices of the longest contiguous lumen run with R in (floor, mouth],
    tolerating short gaps. Avoids the open-mouth/bulk regions (R>mouth) and
    wall-piercing regions (R<=floor)."""
    good = (R > floor) & (R <= mouth)
    idx = np.where(good)[0]
    if len(idx) == 0:
        good = R > floor
        idx = np.where(good)[0]
    bridged = good.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            bridged[a:b] = True
    idx = np.where(bridged)[0]
    if len(idx) == 0:
        return np.array([int(np.argmax(R))])
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1)
    return max(runs, key=len)


def cylinder_axis(coords, origin):
    """Best-fit axis of a (hollow) cylinder: the direction minimising the
    variance of atoms' perpendicular distance to the axis. Robust for barrels
    whose pore is offset from the centroid (e.g. VDAC's N-terminal helix)."""
    X = coords - origin

    def var_perp(angles):
        th, ph = angles
        d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        s = X @ d
        rp = np.linalg.norm(X - np.outer(s, d), axis=1)
        return rp.var()

    best, best_v = None, np.inf
    for th0 in np.linspace(0, np.pi, 7):
        for ph0 in np.linspace(0, np.pi, 7):
            res = minimize(var_perp, [th0, ph0], method="Nelder-Mead",
                           options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 200})
            if res.fun < best_v:
                best_v, best = res.fun, res.x
    th, ph = best
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def find_pore_axis(coords, radii, origin, seed_dir, n_slices=45,
                   tilt_deg=22, n_tilt=5):
    """Search directions near `seed_dir` for the one giving the longest, most
    open enclosed lumen. Returns best direction (unit)."""
    seed = np.asarray(seed_dir, float); seed /= np.linalg.norm(seed)
    u, v = _ortho(seed)
    tilts = np.linspace(-np.deg2rad(tilt_deg), np.deg2rad(tilt_deg), n_tilt)
    best, best_score = None, -1
    for tu in tilts:
        for tv in tilts:
            d = seed + np.tan(tu) * u + np.tan(tv) * v
            d /= np.linalg.norm(d)
            zs, cen, R, _, _ = profile_along(d, origin, coords, radii,
                                             n_slices=n_slices, n_grid=13,
                                             polish=False)
            run = _longest_interior_run(R)
            if len(run) < 3:
                continue
            span = zs[run[-1]] - zs[run[0]]
            # reward a long lumen that stays open (positive min radius)
            score = span * max(R[run].min(), 0.0)
            if score > best_score:
                best_score, best = score, d
    return best if best is not None else seed


def pore_profile(coords, radii, n_slices=200, axis_choice="auto"):
    """Compute the channel axis, centerline and radius profile.

    Returns dict with arrays along the data-derived lumen.
    """
    center0, axes, spread = principal_axes(coords)

    if axis_choice == "auto":
        # subsample atoms for the (cheap) axis search; the pore axis may align
        # with any principal axis (tall channels -> largest extent; squat
        # barrels -> smallest), so seed from all three and refine each locally.
        rng = np.random.default_rng(0)
        if len(coords) > 3500:
            sel = rng.choice(len(coords), 3500, replace=False)
            cs, rs = coords[sel], radii[sel]
        else:
            cs, rs = coords, radii
        seeds = [axes[:, ai] for ai in range(3)]
        seeds.append(cylinder_axis(cs, center0))
        candidates = []
        for d0 in seeds:
            d = find_pore_axis(cs, rs, center0, d0, n_slices=45, n_tilt=5)
            zs, cen, R, u, v = profile_along(d, center0, cs, rs,
                                             n_slices=60, n_grid=13, polish=False)
            run = _longest_interior_run(R)
            span = zs[run[-1]] - zs[run[0]] if len(run) > 1 else 0.0
            score = span * max(R[run].min(), 0.0) if len(run) > 2 else 0.0
            candidates.append((score, d))
        d = max(candidates, key=lambda t: t[0])[1]
    else:
        d = axes[:, int(axis_choice)]
        d = find_pore_axis(coords, radii, center0, d)

    zs, pts_all, R_all, u_axis, v_axis = profile_along(
        d, center0, coords, radii, n_slices=n_slices, n_grid=21)
    w_axis = d

    run = _longest_interior_run(R_all)
    z = zs[run]
    R = R_all[run]
    pts = pts_all[run]

    # centerline tortuosity from a smoothed centerline (raw per-slice centres
    # carry sub-Angstrom discretisation jitter that would inflate arc length).
    def smooth(a, w=5):
        if len(a) < w:
            return a
        k = np.ones(w) / w
        return np.array([np.convolve(a[:, j], k, mode="same") for j in range(a.shape[1])]).T
    pts_s = smooth(pts)
    pts_s[:2], pts_s[-2:] = pts[:2], pts[-2:]  # keep endpoints honest
    seg = np.linalg.norm(np.diff(pts_s, axis=0), axis=1)
    arc = seg.sum()
    axial = abs(z[-1] - z[0])
    tau = arc / axial if axial > 0 else 1.0

    bi = int(np.argmin(R))

    # cross-sectional anisotropy at the bottleneck: directional clearance
    aniso = directional_clearance(pts[bi], u_axis, v_axis, coords, radii)

    return dict(
        axis_index=-1,
        axis=w_axis.tolist(), u=u_axis.tolist(), v=v_axis.tolist(),
        origin=center0.tolist(),
        z_A=z.tolist(),
        centerline_A=pts.tolist(),
        R_A=R.tolist(),
        R_nm=(R / 10.0).tolist(),
        z_nm=((z - z[0]) / 10.0).tolist(),
        length_nm=float((z[-1] - z[0]) / 10.0),
        arc_length_nm=float(arc / 10.0),
        tortuosity=float(tau),
        bottleneck_R_nm=float(R[bi] / 10.0),
        bottleneck_z_nm=float((z[bi] - z[0]) / 10.0),
        bottleneck_center=pts[bi].tolist(),
        mean_R_nm=float(R.mean() / 10.0),
        aniso_min_nm=float(aniso[0] / 10.0),
        aniso_max_nm=float(aniso[1] / 10.0),
        aniso_ratio=float(aniso[1] / aniso[0]) if aniso[0] > 0 else 1.0,
    )


def directional_clearance(center, u_axis, v_axis, coords, radii, n_ang=72):
    """Clearance from center to wall along each in-plane direction; returns
    (min, max) wall distance (Angstrom). Detects slit-like cross-sections."""
    angs = np.linspace(0, np.pi, n_ang, endpoint=False)  # opposite dirs symmetric
    clear = []
    for a in angs:
        dirn = np.cos(a) * np.asarray(u_axis) + np.sin(a) * np.asarray(v_axis)
        # closest atom surface along +/- dirn within a thin transverse slab
        d = coords - center
        s = d @ dirn
        perp = d - np.outer(s, dirn)
        perp_d = np.linalg.norm(perp, axis=1)
        # consider atoms roughly on this line (within their own radius of it)
        m = perp_d < radii + 0.5
        if not m.any():
            clear.append(np.inf)
            continue
        # nearest wall in either +/- direction
        dist = np.abs(s[m]) - radii[m]
        dpos = dist[s[m] >= 0]
        dneg = dist[s[m] < 0]
        rpos = dpos.min() if len(dpos) else np.inf
        rneg = dneg.min() if len(dneg) else np.inf
        clear.append(min(rpos, rneg))
    clear = np.array(clear)
    clear = clear[np.isfinite(clear)]
    if len(clear) == 0:
        return (0.0, 0.0)
    return (float(clear.min()), float(clear.max()))


if __name__ == "__main__":
    import json, sys
    DATA = "data/"
    channels = {
        "7AHL": ("7ahl.cif", "alpha-Hemolysin (heptamer)"),
        "3EMN": ("3emn.cif", "VDAC1 beta-barrel porin"),
        "2ZW3": ("2zw3.cif", "Connexin-26 connexon"),
        "MspA": ("1uun_bio.cif", "MspA octameric porin"),
    }
    out = {}
    for pid, (fn, desc) in channels.items():
        coords, radii = load_atoms(DATA + fn)
        prof = pore_profile(coords, radii)
        prof["name"] = desc
        prof["pdb"] = fn.split(".")[0].upper().replace("_BIO", "")
        prof["n_atoms"] = int(len(coords))
        out[pid] = prof
        print(f"{pid:6s} {desc:28s} "
              f"R_bot={prof['bottleneck_R_nm']*10:5.2f} A "
              f"(d={prof['bottleneck_R_nm']*20:4.1f} A) "
              f"L={prof['length_nm']:4.1f} nm  tau={prof['tortuosity']:.3f}  "
              f"aniso={prof['aniso_ratio']:.2f}")
    with open("results/channel_geometry.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/channel_geometry.json")
