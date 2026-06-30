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
nanorod, carrying **9.3×–19.4×** the cargo of the largest passing sphere — a
gain confirmed independently by atomistic collision detection, dynamically
accessible (Brownian dynamics), and scale-invariant up to the mesoscale barriers
relevant to drug delivery.

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
