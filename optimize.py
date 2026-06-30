"""
optimize.py -- Shape optimisation of rigid particles for channel transport.

Particle family: the two-exponent superquadric (Barr 1981; Jaklic et al. 2000)
    ( |x/rx|^(2/e2) + |y/ry|^(2/e2) )^(e2/e1) + |z/c|^(2/e1) <= 1
with the long (transport) axis along z (half-length c) and transverse
half-widths rx, ry. e1 controls axial squareness (e1=1 ellipsoid end-caps,
e1->0 flat-ended cylinder), e2 controls transverse squareness (e2=1 round
cross-section, e2->0 square). Volume (closed form, MC-verified):
    V = 2 rx ry c e1 e2 B(e1/2+1, e1) B(e2/2, e2/2)

The transport problem: maximise transportable volume subject to
  (i) the transverse cross-section fitting through the channel aperture, and
  (ii) the half-length c being navigable (aspect-ratio / moving-sofa bound).

This module is unit-agnostic; we pass radii in nm.
"""
import numpy as np
from scipy.special import beta as Bfun

EPS_MIN, EPS_MAX = 0.20, 1.0   # avoid numerically singular pinched shapes


def sq_volume(rx, ry, c, e1, e2):
    return 2.0 * rx * ry * c * e1 * e2 * Bfun(e1 / 2 + 1, e1) * Bfun(e2 / 2, e2 / 2)


def superellipse_boundary(rx, ry, m, n=240):
    """Boundary points of |x/rx|^m + |y/ry|^m = 1."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = rx * np.sign(ct) * np.abs(ct) ** (2.0 / m)
    y = ry * np.sign(st) * np.abs(st) ** (2.0 / m)
    return x, y


def circumradius(rx, ry, m):
    """Max distance from centre to the (rx,ry,exp m) superellipse boundary."""
    x, y = superellipse_boundary(rx, ry, m)
    return np.sqrt(x * x + y * y).max()


def fit_in_circle(R_ap, e2, aspect=1.0):
    """Largest (rx,ry) superellipse (transverse exponent m=2/e2, ry/rx=aspect)
    fitting inside a circular aperture of radius R_ap. Returns (rx, ry)."""
    m = 2.0 / e2
    rx0, ry0 = 1.0, aspect
    cr = circumradius(rx0, ry0, m)
    s = R_ap / cr
    return rx0 * s, ry0 * s


def fit_in_ellipse(p, q, e2, aspect=1.0):
    """Largest (rx,ry) superellipse (m=2/e2, ry/rx=aspect) inside an elliptical
    aperture with half-axes (p,q). Maximises area by binary search on scale."""
    m = 2.0 / e2
    bx, by = superellipse_boundary(1.0, aspect, m)
    lo, hi = 0.0, 5.0 * max(p, q)
    for _ in range(60):
        s = 0.5 * (lo + hi)
        if np.all((s * bx / p) ** 2 + (s * by / q) ** 2 <= 1.0):
            lo = s
        else:
            hi = s
    return lo, lo * aspect


def optimise_particle(aperture, AR_max, shell=0.0, ne=9, naspect=9,
                      anisotropic=False):
    """Maximise transportable volume.

    aperture: ('circle', R) or ('ellipse', p, q)  [nm]
    AR_max:   max allowed aspect ratio c / min(rx,ry)  (navigability cap)
    shell:    wall thickness subtracted from each semi-axis for cargo volume

    Returns dict of optimal shape + volumes.
    """
    e1s = np.linspace(EPS_MIN, EPS_MAX, ne)
    e2s = np.linspace(EPS_MIN, EPS_MAX, ne)
    aspects = np.linspace(1.0, 4.0, naspect) if anisotropic else [1.0]
    best = None
    for e1 in e1s:
        for e2 in e2s:
            for asp in aspects:
                if aperture[0] == "circle":
                    rx, ry = fit_in_circle(aperture[1], e2, asp)
                else:
                    rx, ry = fit_in_ellipse(aperture[1], aperture[2], e2, asp)
                r_small = min(rx, ry)
                c = AR_max * r_small
                V = sq_volume(rx, ry, c, e1, e2)
                # cargo volume after subtracting a shell from each semi-axis
                rxc, ryc, cc = rx - shell, ry - shell, c - shell
                Vc = sq_volume(rxc, ryc, cc, e1, e2) if min(rxc, ryc, cc) > 0 else 0.0
                if best is None or V > best["V"]:
                    best = dict(rx=rx, ry=ry, c=c, e1=e1, e2=e2, aspect=asp,
                                V=V, Vcargo=Vc, AR=c / r_small)
    return best


def optimise_navigable(aperture, rho_nm, L_nm, shell=0.0, ne=9, naspect=9,
                       anisotropic=False):
    """Curvature-coupled optimisation.

    A particle that fills the aperture (sigma=1) has no clearance to turn, so in
    a bent channel (finite centreline radius rho) its navigable half-length ->0.
    Shrinking the cross-section by sigma in (0,1] opens a turning clearance
    (1-sigma)*q (q = narrow aperture half-axis) and lengthens the navigable
    segment via the ladder-in-a-bent-tube bound a_max = 2 sqrt(rho * clearance),
    capped by the channel half-length L/2. We optimise this trade-off.
    """
    q = aperture[1] if aperture[0] == "circle" else min(aperture[1], aperture[2])
    e1s = np.linspace(EPS_MIN, EPS_MAX, ne)
    e2s = np.linspace(EPS_MIN, EPS_MAX, ne)
    aspects = np.linspace(1.0, 4.0, naspect) if anisotropic else [1.0]
    sigmas = np.linspace(0.15, 1.0, 24)
    best = None
    for e1 in e1s:
        for e2 in e2s:
            for asp in aspects:
                if aperture[0] == "circle":
                    rx0, ry0 = fit_in_circle(aperture[1], e2, asp)
                else:
                    rx0, ry0 = fit_in_ellipse(aperture[1], aperture[2], e2, asp)
                for sg in sigmas:
                    rx, ry = sg * rx0, sg * ry0
                    clear = (1.0 - sg) * q
                    if np.isinf(rho_nm):
                        a_max = L_nm / 2.0
                    else:
                        a_max = min(L_nm / 2.0, 2.0 * np.sqrt(max(rho_nm * clear, 0.0)))
                    c = a_max
                    if c <= 0:
                        continue
                    V = sq_volume(rx, ry, c, e1, e2)
                    rxc, ryc, cc = rx - shell, ry - shell, c - shell
                    Vc = sq_volume(rxc, ryc, cc, e1, e2) if min(rxc, ryc, cc) > 0 else 0.0
                    if best is None or V > best["V"]:
                        best = dict(rx=rx, ry=ry, c=c, e1=e1, e2=e2, aspect=asp,
                                    sigma=sg, V=V, Vcargo=Vc,
                                    AR=c / min(rx, ry))
    return best


def sphere_baseline(aperture, shell=0.0, L_nm=None):
    """Largest sphere that BOTH passes the aperture and fits the barrier length
    (a sphere of radius R cannot traverse a barrier shorter than 2R)."""
    R = aperture[1] if aperture[0] == "circle" else min(aperture[1], aperture[2])
    if L_nm is not None:
        R = min(R, L_nm / 2.0)
    V = 4.0 / 3.0 * np.pi * R ** 3
    Rc = R - shell
    Vc = 4.0 / 3.0 * np.pi * Rc ** 3 if Rc > 0 else 0.0
    return dict(R=R, V=V, Vcargo=Vc)


# ---- navigability: aspect-ratio caps from channel geometry ----

def ar_cap_length(L_nm, R_bot_nm):
    """Aspect cap if the half-length is limited by the channel half-length."""
    return (L_nm / 2.0) / R_bot_nm


def curvature_radius_from_tortuosity(tau, L_arc_nm):
    """Estimate centreline radius of curvature for a circular-arc model:
    tau = (theta/2) / sin(theta/2);  rho = L_arc / theta."""
    if tau <= 1.0001:
        return np.inf
    from scipy.optimize import brentq
    f = lambda phi: (phi / np.sin(phi)) - tau
    phi = brentq(f, 1e-3, np.pi - 1e-3)
    theta = 2 * phi
    return L_arc_nm / theta


def ar_cap_curvature(tau, L_arc_nm, R_eff_nm, R_bot_nm):
    """Moving-sofa / ladder-in-a-bent-tube cap: longest straight segment that
    fits in a tube of clearance R_eff whose centreline has radius rho:
        a_max = 2 sqrt(rho * R_eff);  AR = a_max / R_bot."""
    rho = curvature_radius_from_tortuosity(tau, L_arc_nm)
    if not np.isfinite(rho):
        return np.inf
    a_max = 2.0 * np.sqrt(rho * R_eff_nm)
    return a_max / R_bot_nm


if __name__ == "__main__":
    import json
    with open("results/channel_geometry.json") as f:
        geo = json.load(f)

    AR_GRID = np.linspace(1.0, 14.0, 53)
    out = {}
    print(f"{'chan':6s} {'R_bot':>6s} {'L':>5s} {'tau':>5s} {'rho':>6s} "
          f"{'gain':>6s} {'AR*':>5s} {'shape':>16s}")
    for pid, g in geo.items():
        R = g["bottleneck_R_nm"]
        L = g["length_nm"]
        tau = g["tortuosity"]
        L_arc = g["arc_length_nm"]
        ap = ("circle", R)
        base = sphere_baseline(ap)

        # gain vs allowed aspect ratio (straight-channel family) for a panel
        gains = [optimise_particle(ap, AR)["V"] / base["V"] for AR in AR_GRID]

        rho = curvature_radius_from_tortuosity(tau, L_arc)   # nm (for reporting)
        # Per-channel geometric potential: the straight-tube optimum
        # (length-capped). The achieved, tortuosity-throttled value is measured
        # atomistically in validate.py; the analytic curvature bound drives the
        # idealised design map (designmap.py), not noisy per-channel tau.
        o = optimise_navigable(ap, np.inf, L)
        gain = o["V"] / base["V"]

        out[pid] = dict(
            name=g["name"], pdb=g["pdb"], R_bot_nm=R, L_nm=L, tortuosity=tau,
            arc_length_nm=L_arc, rho_nm=(None if np.isinf(rho) else float(rho)),
            aniso_ratio=g["aniso_ratio"],
            sphere_V_nm3=base["V"], sphere_R_nm=base["R"],
            AR_grid=AR_GRID.tolist(), gain_grid=gains,
            AR_cap_length=ar_cap_length(L, R),
            AR_cap_curvature=float(ar_cap_curvature(tau, L_arc, 0.5 * R, R)),
            opt_at_length=o, gain_at_length=gain,
        )
        rho_s = "inf" if np.isinf(rho) else f"{rho:5.1f}"
        print(f"{pid:6s} {R:6.2f} {L:5.1f} {tau:5.2f} {rho_s:>6s} "
              f"{gain:5.1f}x {o['AR']:4.1f}  "
              f"e1={o['e1']:.2f} e2={o['e2']:.2f} sig={o['sigma']:.2f}")

    with open("results/optimization.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/optimization.json")
