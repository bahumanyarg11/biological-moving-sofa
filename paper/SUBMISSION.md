# Submission package — Physical Review E

## Decision
- **Target:** Physical Review E (APS). Q1, free on the standard route, manuscript
  already in the correct format (`revtex4-2 [aps,pre]`).
- **Backup if rejected:** J. Chem. Phys. (Q1, RevTeX) → European Physical
  Journal E (Q3, highest acceptance, perfect scope) → Soft Matter (RSC).
- JCIM was considered and rejected: higher IF (~6.4, Q1) but scope is
  cheminformatics/molecular-modeling — desk-reject risk for this soft-matter
  geometry paper.

## 3 things to fill before you hit submit
1. In `main.tex`: replace `[Author Name]` and `[Department, Institution, City,
   Country]` with real values; add ORCID if you have one.
2. In `cover_letter.txt`: fill the bracketed fields (name, affiliation, date,
   referee names).
3. Pick 3–5 suggested referees (candidates below).

## How to submit (APS)
1. Go to the APS submission system (journals.aps.org → "Submit").
2. Choose **Physical Review E → Regular Article**.
3. Upload `main.tex` + `references.bib` + all figures from `../figs/`
   (`Figure1_concept.pdf` … `Figure9_validation.pdf`). APS compiles RevTeX
   server-side; you can also upload the compiled `main.pdf`.
4. Paste the abstract; add PACS/keywords (suggestions below).
5. Upload `cover_letter.txt` content as the cover letter.
6. Enter suggested/opposed referees.
7. Confirm: original, not under consideration elsewhere, no competing interests.

## Suggested referee profiles (fill with specific names from the cited literature)
- A nanoparticle-shape / drug-delivery modeler (e.g. authors of refs on
  non-spherical carriers, filomicelles, margination).
- An ion-channel / nanopore geometry expert (e.g. HOLE / nanopore-sensing
  community).
- A soft-matter transport / entropic-barrier theorist (e.g. Fick–Jacobs,
  translocation modeling).
- A computational biophysicist working on the nuclear pore complex / crowding.

## Keywords
nanoparticle shape; channel transport; moving-sofa problem; superquadric;
entropic barrier; translocation; nuclear pore; drug delivery; pore geometry.

## Strongly recommended before/at submission
- Deposit `v2/` code + derived data on GitHub and archive on Zenodo to obtain a
  DOI; update the Data Availability statement from "on request" to that DOI.
- Optionally add a Supplementary Information PDF (volume-law verification table,
  full parameter and PMF tables).
