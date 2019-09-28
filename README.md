From `binder` using `nbgitpuller`
[![Binder](https://mybinder.org/badge_logo.svg)](https://gke.mybinder.org/v2/gh/mathematicalmichael/thesis/binder?urlpath=git-pull?repo=https://github.com/mathematicalmichael/thesis)

---

# Michael Pilosov's Dissertation

Template based on [this repository](github.com/dewittpe/ucd-dissertation-template).

## Building

Once cloned, run `make dissertation.tex` to build the PDF.

Alternatively, click the binder badge above, which will launch a Jupyterlab interface that has [jupyterlab-latex](https://github.com/jupyterlab/jupyterlab-latex) pre-installed, so you can right-click on the opened `dissertation.tex` file to `Show LaTeX Preview` (as per usual, it may take several builds for all citations and references to link properly).

If using [Atom](https://atom.io/), here are the packages I installed in order to get things working (make sure the packages in `/binder/apt.txt` are also installed on your computer first since these are the LaTeX dependencies on which we rely).
  - atom-latex [0.8.10]
  -~~language-latex [1.2.0]~~

The following were simply for visual purposes (italics = highly recommend):
  - *minimap*
  - *minimap-highlight-selected*
  - minimap-selection
  - minimap-autohide
  - minimap-lens
  - *highlight-selected*
  - highlight-line
  - *multi-cursor*
