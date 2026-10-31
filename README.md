# The biological moving-sofa problem

Shape optimization of rigid nanoparticles for transport through molecular
channels — code, data and manuscript.

This repository accompanies the manuscript *"The biological moving-sofa problem:
shape optimization of rigid nanoparticles for transport through molecular
channels"* (R. G. Bahumanya). It casts rigid-particle passage through a
biological channel as a three-dimensional version of Moser's moving-sofa
problem and solves it on atomistically measured channel geometry.

**Key result.** Across four canonical biological nanopores (α-hemolysin, VDAC1,
connexin-26, MspA) the optimal transporter is not a sphere but a slender
nanorod as long as the channel, carrying **6.6×–13.8×** the cargo of the largest
passing sphere (α-hemolysin 13.8×, connexin-26 11.4×, VDAC1 7.2×, MspA 6.6×). A
rod spanning the full channel length fits each lumen and slides straight through
it without atomic clashes, and the gain is scale-invariant up to the mesoscale
barriers relevant to drug delivery.

## Correction (31 Oct 2026)

Earlier versions of `validate.py` grew the test rod's half-length on a grid
that ended at 0.7·L, so "validated" rods could be up to 1.4× the channel length,
with their ends in bulk solvent where no clash is possible. In all four channels
the reported navigable length was simply the last grid point, which inflated the
validated aspect ratios (13.3 / 6.9 / 10.9 / 6.4) and gains (19.4 / 10.0 / 15.9 /
9.3×, quoted as 9.3–19.4×). The rod half-length is now capped at the channel
half-length L/2, the same cap used in `optimize.py`, and the rod must lie inside
the channel span. All four rods reach that cap without clashes, giving aspect
ratios 9.5 / 4.9 / 7.8 / 4.5 and gains 13.8 / 7.2 / 11.4 / 6.6× (α-hemolysin /
VDAC1 / connexin-26 / MspA). `results/validation.json`, `results/bd.json`,
Figures 4–7 and 9, the TOC graphic and the manuscript have been regenerated. The
Brownian-dynamics sphere-versus-rod barrier comparison turned out to be below the
resolution of the Monte-Carlo sampling (588 configurations per slice), so the
manuscript no longer claims that rods incur no barrier penalty; the disc-jamming
result is unchanged.

## Repository layout

```
data/         Experimental structures (PDB: 7AHL, 3EMN, 2ZW3, 1UUN biol. assembly)
geom.py       HOLE-style inscribed-sphere lumen profiler
optimize.py   Superquadric optimizer (Monte-Carlo-verified volume law)
validate.py   Atomistic collision validation
bd.py         Brownian-dynamics translocation (entropic PMF + Langevin)
mesoscale.py  Scale-up to biological barriers (NPC, mucus, ECM, fenestrae, EPR)
designmap.py  Navigability design map
figstyle.py / molsurface.py / vtkrender.py / figures.py   Figure generation
results/      Derived data (JSON)
figs/         Publication figures (PDF + 600-dpi PNG)
paper/        LaTeX source (RevTeX/PRE), references, compiled PDF
```

## Reproduce

```bash
pip install -r requirements.txt
python geom.py         # lumen geometry  -> results/channel_geometry.json
python optimize.py     # shape optimization
python validate.py     # collision validation
python bd.py           # Brownian-dynamics translocation
python mesoscale.py    # mesoscale extension
python designmap.py    # design map
python vtkrender.py    # molecular renders
python figures.py      # all figures
```

Compile the paper:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data

Atomic coordinates are from the RCSB Protein Data Bank (entries 7AHL, 3EMN,
2ZW3, 1UUN) and are redistributed here under the PDB's open terms for
reproducibility.

## Citation

If you use this work, please cite the manuscript (under review). BibTeX will be
added upon publication.

## License

Code released under the MIT License (see `LICENSE`).
