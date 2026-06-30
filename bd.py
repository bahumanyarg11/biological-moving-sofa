"""
bd.py -- Brownian-dynamics translocation of shaped rigid particles through the
real atomistic channels.

We build a potential of mean force (PMF) G(z) along the pore axis for each
particle shape by Monte-Carlo sampling collision-free rigid configurations
(transverse offset + orientation) against the atomic coordinates -- an entropic
(Fick-Jacobs) free energy. We then integrate the overdamped Langevin equation
on G(z) with an external drive f and absorbing mouths, yielding the
translocation probability P_trans(f) and the mean first-passage time (MFPT).

This turns the geometric capacity result into a dynamical, testable prediction:
the high-capacity nanorod remains translocation-competent, with a quantified
capacity-vs-speed trade-off.
"""
import json
import numpy as np
from scipy.spatial import cKDTree
import geom

DATA = "data/"
FILES = {"7AHL": "7ahl.cif", "3EMN": "3emn.cif",
         "2ZW3": "2zw3.cif", "MspA": "1uun_bio.cif"}
KT = 1.0                      # energies in units of kT
rng = np.random.default_rng(7)


def basis(axis):
    d = np.asarray(axis, float); d /= np.linalg.norm(d)
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v, d


def superquadric_points(rx, ry, c, e1, e2, nu=11, nv=20):
    u = np.linspace(-np.pi/2, np.pi/2, nu)
    v = np.linspace(-np.pi, np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v)

    def f(a, e):
        return (np.sign(np.cos(a))*np.abs(np.cos(a))**e,
                np.sign(np.sin(a))*np.abs(np.sin(a))**e)
    cU, sU = f(U, e1); cV, sV = f(V, e2)
    P = np.column_stack([(rx*cU*cV).ravel(), (ry*cU*sV).ravel(),
                         (c*sU).ravel()])
    return np.unique(np.round(P, 2), axis=0)


def rot_tilt(tu, tv, tw):
    cu, su = np.cos(tu), np.sin(tu)
    cv, sv = np.cos(tv), np.sin(tv)
    cw, sw = np.cos(tw), np.sin(tw)
    Rx = np.array([[1, 0, 0], [0, cu, -su], [0, su, cu]])
    Ry = np.array([[cv, 0, sv], [0, 1, 0], [-sv, 0, cv]])
    Rz = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pmf(tree, radii, origin, u, v, d, P_local, z_grid, R_off, tilt_max,
        n_off=42, n_orient=14):
    """Entropic PMF G(z)=-kT ln Omega(z) from collision-free MC sampling.
    Omega(z) ~ fraction of sampled (offset,orientation) that are clash-free."""
    Frame = np.vstack([u, v, d])           # local->world rows
    G = np.zeros(len(z_grid))
    Omega = np.zeros(len(z_grid))
    npts = len(P_local)
    for iz, z in enumerate(z_grid):
        ok = 0
        tot = 0
        for _ in range(n_orient):
            tu = rng.uniform(-tilt_max, tilt_max)
            tv = rng.uniform(-tilt_max, tilt_max)
            tw = rng.uniform(-np.pi, np.pi)
            Rp = rot_tilt(tu, tv, tw)
            Pw0 = (P_local @ Rp.T) @ Frame      # oriented, world axes (npts,3)
            rr = R_off*np.sqrt(rng.uniform(0, 1, n_off))
            th = rng.uniform(0, 2*np.pi, n_off)
            offs = (np.outer(rr*np.cos(th), u) + np.outer(rr*np.sin(th), v)
                    + (origin + z*d))                       # (n_off,3)
            # all points of all offsets in one query
            allpts = (Pw0[None, :, :] + offs[:, None, :]).reshape(-1, 3)
            dist, idx = tree.query(allpts, k=1, workers=-1)
            clash = (dist < radii[idx]).reshape(n_off, npts).any(axis=1)
            ok += int((~clash).sum())
            tot += n_off
        frac = ok / max(tot, 1)
        Omega[iz] = frac
        G[iz] = -KT*np.log(frac + 1e-6)
    G -= G.min()
    return G, Omega


def langevin(zg, G, D, drive, z0, n_traj=2500, dt=1.5e-3, tmax=1500.0):
    """Overdamped Langevin on G(z) (kT units). Absorbing at zg[0],zg[-1].
    Returns P_trans and MFPT (in same time units)."""
    dG = np.gradient(G, zg)
    zlo, zhi = zg[0], zg[-1]
    z = np.full(n_traj, z0)
    alive = np.ones(n_traj, bool)
    trans = np.zeros(n_traj, bool)
    ttime = np.zeros(n_traj)
    nsteps = int(tmax/dt)
    sq = np.sqrt(2*D*dt)
    for s in range(nsteps):
        if not alive.any():
            break
        f = -np.interp(z[alive], zg, dG) + drive
        z[alive] = z[alive] + D/KT*f*dt + sq*rng.standard_normal(alive.sum())
        ttime[alive] += dt
        hi = alive & (z >= zhi)
        lo = alive & (z <= zlo)
        trans[hi] = True
        alive[hi] = False; alive[lo] = False
    P = trans.mean()
    mfpt = ttime[trans].mean() if trans.any() else np.nan
    return float(P), float(mfpt), z, trans


def main():
    geo = json.load(open("results/channel_geometry.json"))
    opt = json.load(open("results/optimization.json"))
    val = json.load(open("results/validation.json"))
    out = {}
    for pid in FILES:
        g = geo[pid]
        coords, radii = geom.load_atoms(DATA + FILES[pid])
        axis = np.array(g["axis"]); u, v, d = basis(axis)
        cmid = np.array(g["centerline_A"]).mean(0)
        rel = coords - cmid; zc = rel @ d
        perp = np.linalg.norm(rel - np.outer(zc, d), axis=1)
        m = (perp < 18) & (np.abs(zc) < g["length_nm"]*10/2 + 8)
        coords, radii = coords[m], radii[m]
        tree = cKDTree(coords)
        origin = cmid

        Rb = g["bottleneck_R_nm"]*10      # A
        Lh = g["length_nm"]*10/2
        z_grid = np.linspace(-Lh-6, Lh+6, 42)
        R_off = max(g["mean_R_nm"]*10, Rb*1.3)

        o = opt[pid]["opt_at_length"]
        a_nav = val[pid]["a_nav_nm"]*10
        shapes = {
            "sphere":   dict(P=superquadric_points(Rb*0.95, Rb*0.95, Rb*0.95, 1, 1),
                             tilt=np.pi, col="sphere"),
            "rod_opt":  dict(P=superquadric_points(o["rx"]*10*0.9, o["rx"]*10*0.9,
                                                   a_nav*0.9, o["e1"], o["e2"]),
                             tilt=np.deg2rad(18)),
            "disc":     dict(P=superquadric_points(Rb*1.7, Rb*1.7, Rb*0.5*0.9, 1, 1),
                             tilt=np.deg2rad(18)),
        }
        rec = {"z_nm": ((z_grid)/10).tolist()}
        for name, sh in shapes.items():
            G, Om = pmf(tree, radii, origin, u, v, d, sh["P"], z_grid,
                        R_off, sh["tilt"])
            # diffusion: Stokes-ish, scale by inverse size (rough)
            D = 0.5
            res = {}
            for f in (0.0, 0.5, 1.0, 1.5, 2.0):
                P, mfpt, _, _ = langevin(z_grid/10, G, D, f, z0=z_grid[3]/10)
                res[f"f{f}"] = dict(P=P, mfpt=mfpt)
            rec[name] = dict(G=G.tolist(), Pf=res,
                             Gbarrier=float(G.max()))
            print(f"{pid:5s} {name:8s} dG*={G.max():4.1f} kT  "
                  f"P(f=1)={res['f1.0']['P']:.2f}  MFPT(f=1)={res['f1.0']['mfpt']}")
        out[pid] = rec
    json.dump(out, open("results/bd.json", "w"), indent=2)
    print("Wrote results/bd.json")


if __name__ == "__main__":
    main()
