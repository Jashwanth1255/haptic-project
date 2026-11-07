# Manuscript

`main.tex` is the paper draft in IEEE conference format. `references.bib` is the
bibliography, and `REFERENCES_CHECKLIST.md` lists what still needs verifying.

## Compile

On Overleaf: create a blank project and upload `main.tex` and `references.bib`.
Make a folder `figures/` and upload the four `.pdf` files from
`../figures/paper/`. Compile with pdfLaTeX.

Locally, with TeX Live or MiKTeX installed:

    cd paper
    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Rules for editing

- Every number comes from a saved run; the run id is in a `%` comment beside it.
  If a number changes, rerun the script and copy the value from the run. Never
  edit a number by hand.
- Regenerate the figures with `python make_paper_figures.py`.
  `figures/paper/SOURCES.json` records which run drew each figure.
- Keep "exploratory", "post hoc" and "pre-registered" exactly where they are. Moving
  them changes what the paper claims.
- No reference goes in until someone has opened its publisher page.

## Before submission

1. Author names, emails and order.
2. Every row of `REFERENCES_CHECKLIST.md` verified; `cao_fabricvst` found or the
   sentence citing it removed.
3. The two `TODO(team)` comments in `main.tex`: the acceleration sample rate
   against the dataset paper, and an archived code release (e.g. a Zenodo DOI).
4. Dataset licence on the figshare record checked against the venue's requirements.
5. Compile, check the page limit for the chosen venue, and read the PDF once
   end to end.
