"""
mesoscale.py -- Scale-up of the shape-optimisation principle to the mesoscale
biological barriers that 10-100 nm therapeutic particles actually encounter.

The navigability gain is dimensionless: it depends only on the achievable
aspect ratio (set by barrier length/tortuosity) and the aperture anisotropy,
NOT on absolute size. Hence the nanorod-beats-sphere result is scale-invariant
from 1 nm porins to ~1 um mucus pores. We (i) demonstrate scale invariance and
(ii) tabulate literature-parameterised gains for the major barriers to delivery.
"""
import json
import numpy as np
from optimize import (optimise_navigable, sphere_baseline,
                      curvature_radius_from_tortuosity)

# Literature-informed, order-of-magnitude descriptors of biological barriers.
# R_bot: constriction radius (nm); L: traversal length (nm); tau: tortuosity;
# eps_ap: aperture anisotropy (1=circular, >1 slit-like).
SYSTEMS = [
    dict(key="TJ",   name="Tight junction (claudin)", R_bot=0.35, L=4.0,
         tau=1.05, eps_ap=2.5, cargo="ions, small drugs", scale=0.7),
    dict(key="CX",   name="Connexin gap junction",     R_bot=0.6,  L=15.0,
         tau=1.1,  eps_ap=1.5, cargo="metabolites, siRNA?", scale=1.2),
    dict(key="NPCp", name="Nuclear pore (passive)",    R_bot=2.5,  L=40.0,
         tau=1.3,  eps_ap=1.2, cargo="<40 kDa proteins", scale=5.0),
    dict(key="NPCf", name="Nuclear pore (facilitated)", R_bot=20.0, L=90.0,
         tau=1.6,  eps_ap=1.2, cargo="10-40 nm RNPs, virus", scale=39.0),
    dict(key="ECM",  name="Collagen / ECM mesh",       R_bot=30.0, L=300.0,
         tau=1.8,  eps_ap=1.6, cargo="20-100 nm NPs", scale=60.0),
    dict(key="MUC",  name="Mucus mesh",                R_bot=100.0, L=800.0,
         tau=2.0,  eps_ap=1.5, cargo="<200 nm NPs", scale=200.0),
    dict(key="FEN",  name="Liver fenestrae",           R_bot=55.0, L=60.0,
         tau=1.05, eps_ap=1.3, cargo="50-150 nm NPs", scale=110.0),
    dict(key="EPR",  name="Tumour vascular gaps (EPR)", R_bot=300.0, L=400.0,
         tau=1.2,  eps_ap=1.4, cargo="100-700 nm NPs", scale=600.0),
]


def gain_for(R_bot, L, tau, eps_ap):
    p = R_bot*np.sqrt(eps_ap); q = R_bot/np.sqrt(eps_ap)
    rho = curvature_radius_from_tortuosity(tau, tau*L)
    o = optimise_navigable(("ellipse", p, q), rho, L, anisotropic=True,
                           ne=7, naspect=9)
    base = sphere_baseline(("ellipse", p, q), L_nm=L)
    return o["V"]/base["V"], o


def main():
    out = {"systems": []}
    print(f"{'system':30s} {'R(nm)':>6s} {'L(nm)':>6s} {'tau':>4s} "
          f"{'eps':>4s} {'gain':>6s} {'AR':>5s}")
    for s in SYSTEMS:
        g, o = gain_for(s["R_bot"], s["L"], s["tau"], s["eps_ap"])
        rec = dict(s)
        rec.update(gain=float(g), AR=float(o["AR"]),
                   e1=float(o["e1"]), e2=float(o["e2"]),
                   sigma=float(o["sigma"]))
        out["systems"].append(rec)
        print(f"{s['name']:30s} {s['R_bot']:6.1f} {s['L']:6.0f} {s['tau']:4.2f} "
              f"{s['eps_ap']:4.1f} {g:5.1f}x {o['AR']:4.1f}")

    # scale-invariance demonstration: identical (AR_max via L/R, eps_ap, tau)
    # across 3 decades of pore size -> identical gain.
    radii = np.logspace(np.log10(0.3), np.log10(1000), 40)   # nm
    LR = 16.0          # fixed length/radius ratio
    inv = {"R_nm": radii.tolist()}
    for tau, eps in [(1.05, 1.0), (1.3, 1.0), (1.05, 2.5), (1.8, 1.0)]:
        gains = []
        for R in radii:
            g, _ = gain_for(R, LR*R, tau, eps)
            gains.append(g)
        inv[f"tau{tau}_eps{eps}"] = gains
    out["scale_invariance"] = inv
    json.dump(out, open("results/mesoscale.json", "w"), indent=2)
    print("\nWrote results/mesoscale.json")


if __name__ == "__main__":
    main()
