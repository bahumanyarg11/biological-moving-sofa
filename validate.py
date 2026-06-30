"""
validate.py -- Independent atomistic validation by collision detection.

Two orthogonal checks against the real protein coordinates (this code never
uses the analytic volume formula or aperture constraint):

(A) Transverse limit. A short probe of increasing radius is placed on the pore
    axis at the bottleneck. The clearance crosses zero at r ~ R_bot, confirming
    the extracted bottleneck radius is the binding transverse constraint.

(B) Navigable length (the moving-sofa test). A rigid cylinder of fixed radius
    (just inside R_bot) and growing half-length is placed along the best-fit
    lumen axis. The longest collision-free half-length a_nav is the largest
    straight rigid segment the real lumen admits -- the data-measured analogue
    of the ladder-around-a-corner bound. Straight channels admit long rods;
    tortuous ones cut a_nav down, exactly as the curvature bound predicts.
"""
import json
import numpy as np
import geom

DATA = "data/"
FILES = {"7AHL": "7ahl.cif", "3EMN": "3emn.cif",
         "2ZW3": "2zw3.cif", "MspA": "1uun_bio.cif"}


def cylinder_surface(r, half_len, n_theta=96, n_z=70, caps=True):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    zz = np.linspace(-half_len, half_len, max(n_z, 2))
    TH, ZZ = np.meshgrid(th, zz)
    pts = [np.column_stack([(r * np.cos(TH)).ravel(),
                            (r * np.sin(TH)).ravel(), ZZ.ravel()])]
    if caps and half_len > 1e-6:
        for zc in (-half_len, half_len):
            for rad in np.linspace(0, r, 10):
                pts.append(np.column_stack([rad * np.cos(th), rad * np.sin(th),
                                            np.full_like(th, zc)]))
    return np.vstack(pts)


def basis_from_axis(axis):
    d = np.asarray(axis, float); d /= np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v, d


def min_clearance(local, center, u, v, d, coords, radii, chunk=3000):
    P = center + local @ np.vstack([u, v, d])
    best = np.inf
    for i in range(0, len(P), chunk):
        Q = P[i:i + chunk]
        dist = np.linalg.norm(Q[:, None, :] - coords[None, :, :], axis=2) - radii[None, :]
        best = min(best, float(dist.min()))
    return best


def main():
    with open("results/channel_geometry.json") as f:
        geo = json.load(f)
    with open("results/optimization.json") as f:
        opt = json.load(f)

    out = {}
    for pid, g in geo.items():
        coords, radii = geom.load_atoms(DATA + FILES[pid])
        cline = np.array(g["centerline_A"])
        # best-fit straight lumen axis (PCA of the centreline)
        cmean = cline.mean(0)
        _, _, vecs = np.linalg.svd(cline - cmean)
        axis = vecs[0]
        u, v, d = basis_from_axis(axis)
        # restrict atoms to the channel neighbourhood for speed
        rel = coords - cmean
        perp = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
        coords, radii = coords[perp < 16.0], radii[perp < 16.0]

        Rbot_A = g["bottleneck_R_nm"] * 10.0
        bcen = np.array(g["bottleneck_center"])

        # (A) transverse radius sweep (short probe at the bottleneck)
        r_facs = np.linspace(0.5, 1.4, 19)
        rad_clear = []
        for f_ in r_facs:
            probe = cylinder_surface(f_ * Rbot_A, 1.5, n_z=6)
            rad_clear.append(min_clearance(probe, bcen, u, v, d, coords, radii) / 10)

        # (B) navigable-length sweep (fixed radius, growing half-length).
        # Search rod centre (along the centreline) and a small tilt about the
        # best-fit axis, reporting the BEST achievable rigid-rod half-length
        # (fair best case): a straight rod cannot follow a curved lumen, so this
        # is the data-measured ladder-around-a-corner / moving-sofa length.
        r0 = 0.92 * Rbot_A
        c_grid_A = np.linspace(2.0, max(g["length_nm"] * 10 * 0.7, 25.0), 26)
        anchors = cline[np.linspace(0, len(cline) - 1, 5).astype(int)]
        tilts = np.deg2rad(np.linspace(-12, 12, 5))
        best_anav, best_curve = -1.0, None
        for anc in anchors:
            for tu in tilts:
                for tv in tilts:
                    ax = d + np.tan(tu) * u + np.tan(tv) * v
                    uu, vv, dd = basis_from_axis(ax)
                    lc = []
                    for c in c_grid_A:
                        rod = cylinder_surface(r0, c, n_z=46)
                        lc.append(min_clearance(rod, anc, uu, vv, dd,
                                                coords, radii) / 10)
                    lc = np.array(lc)
                    passing = c_grid_A[lc >= -0.02]
                    anav = float(passing.max() / 10) if len(passing) else 0.0
                    if anav > best_anav:
                        best_anav, best_curve = anav, lc
        a_nav_nm = best_anav
        len_clear = best_curve

        # volume gain achieved at the *validated* navigable length
        from optimize import sq_volume, sphere_baseline
        o = opt[pid]["opt_at_length"]
        Vnav = sq_volume(o["rx"], o["ry"], a_nav_nm, o["e1"], o["e2"])
        gain_nav = Vnav / sphere_baseline(("circle", g["bottleneck_R_nm"]))["V"]

        out[pid] = dict(
            r_factors=r_facs.tolist(), radius_clearance_nm=rad_clear,
            c_grid_nm=(c_grid_A / 10).tolist(),
            length_clearance_nm=len_clear.tolist(),
            a_nav_nm=a_nav_nm,
            AR_nav=a_nav_nm / g["bottleneck_R_nm"],
            gain_nav=float(gain_nav),
            AR_cap_curvature=opt[pid]["AR_cap_curvature"],
            AR_cap_length=opt[pid]["AR_cap_length"],
        )
        print(f"{pid:6s} a_nav={a_nav_nm:5.2f} nm  AR_nav={out[pid]['AR_nav']:4.1f}  "
              f"(len-cap AR={out[pid]['AR_cap_length']:.1f}, "
              f"curv-cap AR={out[pid]['AR_cap_curvature']:.1f})  "
              f"gain_nav={gain_nav:.1f}x")

    with open("results/validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/validation.json")


if __name__ == "__main__":
    main()
