"""
designmap.py -- Navigability/shape design map.

Optimal transportable-volume gain (vs the largest passing sphere) as a function
of the two direct geometric levers a designer controls:
  - achievable aspect ratio AR = a / R_bot  (set by channel length & tortuosity
    via the moving-sofa bound; see optimize.ar_cap_* and Fig. 6b), and
  - aperture anisotropy eps_ap = p/q  (lets a flattened particle exploit the
    wide aperture direction).
The result is a smooth landscape; the four channels are overlaid at their
atomistically-validated (AR, anisotropy).
"""
import json
import numpy as np
from optimize import optimise_particle, sphere_baseline

R_BOT = 0.50      # nm, representative (gain is scale-invariant)


def gain_at(AR, eps_ap):
    p = R_BOT * np.sqrt(eps_ap)        # area-preserving elliptical aperture
    q = R_BOT / np.sqrt(eps_ap)
    ap = ("ellipse", p, q)
    o = optimise_particle(ap, AR, anisotropic=True, ne=7, naspect=9)
    base = sphere_baseline(ap)         # sphere limited by narrow half-axis q
    return o["V"] / base["V"]


def main():
    ARs = np.linspace(1.0, 14.0, 32)
    epss = np.linspace(1.0, 4.0, 28)
    Z = np.zeros((len(epss), len(ARs)))
    for i, e in enumerate(epss):
        for j, a in enumerate(ARs):
            Z[i, j] = gain_at(a, e)
    out = dict(AR=ARs.tolist(), eps_ap=epss.tolist(), gain=Z.tolist(),
               R_bot_nm=R_BOT)
    with open("results/designmap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("design map gain range: %.2f - %.2f x" % (Z.min(), Z.max()))
    print("Wrote results/designmap.json")


if __name__ == "__main__":
    main()
