# Michael Pilosov's Dissertation

## Getting Started

Launch an interactive editor (using [Binder](https://mybinder.org)): [![Binder](https://mybinder.org/badge_logo.svg)](https://gke.mybinder.org/v2/gh/mathematicalmichael/thesis/binder?urlpath=git-pull?repo=https://github.com/mathematicalmichael/thesis)

> This button/link is pointing to the `binder` branch of this repository and using `nbgitpuller` to sync latest changes from `master` branch. These choices were made to expedite load times, since mybinder.org re-builds the environment if it detects commits. The [`binder` branch](https://github.com/mathematicalmichael/thesis/tree/binder) is an orphan, containing only the build files. The environment contains all the dependencies required to generate this dissertation.

When it launches:
1. Change the web-address ending `/tree/thesis` to `/lab`
1. Open the Terminal by clicking on the icon
1. Copy/Paste the following commands:
```sh
cd thesis; make; make clean
```

> You can then use the File-Browser on the left-hand side to navigate to `thesis/dissertation.pdf` and open a preview PDF in the JupyterLab web-app.

This will run the [makefile](https://github.com/mathematicalmichael/thesis/blob/master/makefile), which calls LaTeXMake to fully compile the thesis with the bibliography and references.

---


Template based on [this repository](github.com/dewittpe/ucd-dissertation-template).

## Building from Source.

Clone the repository, `cd thesis`, and execute `make; make clean` to build the PDF, then open with your favorite viewer (a web-browser works). 

Note that `make` [defaults to](https://github.com/mathematicalmichael/thesis/blob/master/makefile#L7) `make dissertation.pdf`.

> Alternatively, click the binder badge above, which will launch a Jupyterlab interface that has [jupyterlab-latex](https://github.com/jupyterlab/jupyterlab-latex) pre-installed, so you can right-click on the opened `dissertation.tex` file to `Show LaTeX Preview` (as per usual, it may take several builds for all citations and references to link properly). 
Using `make` in Terminal is the _proper_ workflow, but `jupyterlab-latex` can technically be configured to run `latekmk` the same way the makefile does. 


## Docker

Docker is a container-orchestration platform which also offers a cloud registry. 
What this means is that if you have Docker installed on your computer (it runs on any platform), you can download a Linux image with all the dependencies required and build the thesis without installing Python or LaTeX on your computer, since it will be built in an container managed by Docker. This technology enables the Binder service above. 

#### Instructions
Clone the repo and `cd` into the folder, then you can build the PDF with:

`docker run --rm -v $(pwd):/tmp --workdir /tmp mathematicalmichael/math-user:thesis make`

##  Integrated Development Environment

Jupyterlab is a good IDE, and launches with the Binder link above. You can run the Docker image to get the same environment locally in your web browser. 
Alternatively, you might choose to use a different IDE. 

Assuming you have a _local LaTeX installation_ (including `latekmk`), you can use an editor to accomplish a very comfortable environment on your local machine in your favorite editor. I personally like [Atom](https://atom.io), so below are instructions to set up a LaTeX editor capable of building this thesis (with two-way SyncTeX):

If using [Atom](https://atom.io/), here are the packages I installed in order to get things working (make sure the packages in `/binder/apt.txt` are also installed on your computer first since these are the LaTeX dependencies on which we rely).
  - atom-latex [0.8.10]

The following were simply for visual purposes (italics = highly recommend):
  - *minimap*
  - *minimap-highlight-selected*
  - minimap-selection
  - minimap-autohide
  - minimap-lens
  - *highlight-selected*
  - highlight-line
  - *multi-cursor*
