# Doctoral Dissertation of Michael Pilosov


## Introduction
This repository holds the LaTeX code for building Dr. Pilosov's dissertation.
There has been an extensive effort made to ensure the reproducibility of all computational results contained in the document as well as the actual compilation of the work itself.

The title of the work is _Computational Advances in Data-Consistent Inversion: Measure-Theoretic Methods for Parameter Estimation_.

The focus of the work is the extension of a framework designed to solve aleotoric (irreducible) uncertainty quantification problems to the solution of epistemic (reducible) uncertainty problems.
It is concerned with framing a problem such that as more data is collected, uncertainty is reduced.

In other words: the question _How does one construct and solve a parameter-estimation problem as a density-estimation problem?_ is posed and addressed in the context of measure-theoretic solutions to stochastic inverse problems_

## TL;DR
We define the notion of Maximal Updated Density (MUD) points, which are the values that maximize an updated density, analogous to how a MAP (Maximum A-Posteriori) point maximizes a posterior density from Bayesian inversion.
Updated densities (measure-theoretic solutions) differ from posteriors in that they are the solution to a different problem which seeks to match the push-forward of the updated density to a specified observed distribution.
It is the selection of a single point from a density that provides the solution to the parameter estimation problem, but the construction of this density is novel.
A QoI map is defined such that when used in the existing framework, the resulting densities exhibit decreasing variance as more data is incorporated into the solution.


## Quickstart
This repo does not contain the source code for the figures. The entire `figures` directory can be removed and recreated with one command (it included here only for convenience and reference).
This is due to the work in [MUD-Examples](https://github.com/mathematicalmichael/mud-examples.git), which depends on the core methods and routines defined in [MUD](https://github.com/mathematicalmichael/mud.git).
As such, presuming you have a LaTeX compiler and Python 3.6+ installed on your system (along with `pip`), you can reproduce the results and compile this PDF as follows:

```sh
$ git clone https://github.com/mathematicalmichael/thesis
$ cd thesis
$ make
$ make clean # (optional cleanup of build files)
```

Or if you have `docker`, this will pull `docker.io/mathematicalmichael/latex`:

```sh
$ git clone https://github.com/mathematicalmichael/thesis
$ cd thesis
$ ./docker_command
```

If you want to recreate all the figures, you can run:

```sh
$ rm -rf figures/ # (optionally wipe figures directory)
$ pip install mud-examples
$ mud_run_all
```

## Reproducibility
The author went to great lengths to ensure that there would be as little friction as possible to recreate the results presented in this dissertation.

Continuous Integration and Deployment pipelines have been set up for the [MUD](https://github.com/mathematicalmichael/mud.git) and [MUD-Examples](https://github.com/mathematicalmichael/mud-examples.git) repositories to ensure the integrity of the computational results.
You will find unit tests and code coverage reports at both of those projects.

Similarly, this repository also checks the validity of the LaTeX compilation through the use of Github Actions (see `.github/workflows/`).

### Docker
A `Dockerfile` can be found in `bin/` as well as executable shell scripts which use emphemeral docker containers to perform the compilation, so by extending your `$PATH` to include `$(pwd)/bin`, you can "trick" your computer into thinking it has all the requisite `TeX`-related software.
If your system is also missing `make`, you can rename `bin/dmake` ("docker make") to `bin/make` or just invoke `dmake` in place of `make`.

Extending your `PATH` and building with `docker`:
```sh
$ export PATH=$(pwd)/bin:$PATH
$ dmake
```

The docker image to build the document is 
`docker.io/mathematicalmichael/latex`
and all the software used to create the simulation results and figures can be found inside of
`docker.io/mathematicalmichael/python:thesis`, which is built from 

### Debian-based distributions
You can find a list of the relevant `apt` packages in `bin/Dockerfile` but at the time of writing, here are the names that were used for builds based on `ubuntu:20.04`:

```
build-essential \
latexmk \
texlive-base \
texlive-fonts-extra \ 
texlive-fonts-recommended \
texlive-latex-base \
texlive-latex-extra \
texlive-science \
```


### Arhcitectures
Everything has been validated on `AMD64` laptops, desktops, and servers extensively, and the CI pipelines run on this architecture.
Furthermore, some testing has validated that everything (including the actual physics simulations) can be successfully reproduced on hardware such as the Raspberry Pi 4 running a 64-bit operating system, which is an `ARM64` device.

`mud_run_all` has been tested on the new `M1` macbooks (also `ARM64`), and the docker images worked under Rosetta emulation but have not been built as native multi-arch images at the time of writing.
Please open an issue if you would like help compiling or reproducing results on such machines. It is possible, just less convenient.
