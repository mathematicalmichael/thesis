![docker](https://github.com/mathematicalmichael/thesis/actions/workflows/docker.yml/badge.svg)
![latex](https://github.com/mathematicalmichael/thesis/actions/workflows/main.yml/badge.svg)
![figures](https://github.com/mathematicalmichael/thesis/actions/workflows/figures.yml/badge.svg)
![examples](https://github.com/mathematicalmichael/thesis/actions/workflows/examples.yml/badge.svg)


# Doctoral Dissertation of Michael Pilosov


## Introduction
This repository holds the LaTeX code for building Dr. Pilosov's dissertation.
There has been an extensive effort made to ensure the reproducibility of all computational results contained in the document as well as the actual compilation of the work itself.

The title of the work is _Computational Advances in Data-Consistent Inversion: Measure-Theoretic Methods for Parameter Estimation_.

The focus of the work is the extension of a framework designed to solve aleatoric (irreducible) uncertainty quantification problems to the solution of epistemic (reducible) uncertainty problems.
It is concerned with framing a problem such that as more data is collected, uncertainty is reduced.

In other words: the question _How does one construct and solve a parameter-estimation problem as a density-estimation problem?_ is posed and addressed in the context of measure-theoretic solutions to stochastic inverse problems.


## TL;DR
We define the notion of Maximal Updated Density (MUD) points, which are the values that maximize an updated density, analogous to how a MAP (Maximum a posteriori) point maximizes a posterior density from Bayesian inversion.
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

Or if you have `docker`, this will pull `docker.io/mathematicalmichael/latex-thesis` (see below for more options involving files in `/bin` or if you want to build the image yourself):

```sh
$ git clone https://github.com/mathematicalmichael/thesis
$ cd thesis
$ ./docker_make
$ ./docker_make clean
```

If you want to recreate all the figures, you can run:

```sh
$ cd examples
$ make
$ make clean
```

(some figures persist in `figures` and the scripts to generate them are there but not as organized as the author would like them to be)

NOTE: `pip install mud-examples && mud_run_all` will create figures for the paper that is being written from this dissertation. As such, the directory structures differ, and there is some duplicate code in this repository that has been migrated to `mud-examples` and further improved. The author will gradually reduce this duplication and ensure that the only code that remains in this repository will be that associated with experiment orchestration (since some figures will differ).


## Reproducibility
The author went to great lengths to ensure that there would be as little friction as possible to recreate the results presented in this dissertation.

Continuous Integration and Deployment pipelines have been set up for the [MUD](https://github.com/mathematicalmichael/mud.git) and [MUD-Examples](https://github.com/mathematicalmichael/mud-examples.git) repositories to ensure the integrity of the computational results.
You will find unit tests and code coverage reports at both of those projects.

Similarly, this repository also checks the validity of the LaTeX compilation through the use of Github Actions (see `.github/workflows/`).


### Docker
A `Dockerfile` can be found in `bin/` as well as executable shell scripts which use ephemeral docker containers to perform the compilation, so by extending your `$PATH` to include `$(pwd)/bin`, you can "trick" your computer into thinking it has all the requisite `TeX`-related software.
If your system is also missing `make`, you can rename `bin/dmake` ("docker make") to `bin/make` or just invoke `dmake` in place of `make`.

Extending your `PATH` and building with `docker`:
```sh
$ export PATH=$(pwd)/bin:$PATH
$ dmake
```

The docker image to build the document is created from `./bin/Dockerfile` and pushed automatically by Github Actions to
`docker.io/mathematicalmichael/latex-thesis` ([link][latex-thesis-hub]).

All the software used to create the simulation results and figures can be used by pulling
`docker.io/mathematicalmichael/python-thesis` ([link][python-thesis-hub]), which is built from `./bin/Dockerfile-conda` and also published continuously.
This image relies on Fenics, a physics simulation software suite which is no longer supported, and so the Python image was based on Python 3.7.6 originally as this was the last supported installation by `conda` as of late 2021.

In late 2022, thanks to the adoption of `micromamba` ([homepage][mamba-site]), the image was able to be updated to include `fenics` with Python 3.10.6 based on a [mamba image][mamba-hub].
In reality, only data-generation really needs to be tied to this requirement, and in the future the data-generation image may be separated from all
the other python script dependencies in order to validate that newer versions of Python still work (assuming that conda-based options eventually stop working).

To prepare for this eventuality, the author put together a [repo](https://github.com/mindthemath/fenics) to build Fenics from source for many Python versions using the [official Docker Python images][python-hub] for debian base-images supporting both ARM and AMD64 architectures.
The images can be found at [docker hub][fenics-hub].

Note: There is another compatible image with `./bin/Dockerfile-python` which relies on the official `current` image from the maintainers of Fenics, but that uses Python 3.6.7.
It is published at `docker.io/mathematicalmichael/python:thesis` (before the author understood how to properly use image tags).


For what it is worth, MUD and MUD-Examples both (at the time of writing, a year after defense), test Python versions up to 3.9.7, so with the exception of data-generation with Fenics for the PDE-based examples, everything else appears to be compatible with newer versions of python (and other architectures, which are discussed below).


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


### Supported Architectures
We are living in a transitional time where low-power ARM devices are starting to become widely used.
During the writing of this dissertation, the author experimented with reproducing his results on some such machines and has made an effort to support their adoption into the reproducibility considerations.
Everything has been validated on `AMD64` laptops, desktops, and servers extensively, and the CI pipelines run on this architecture.
This was the platform that was primarily used for the development of this work.
Furthermore, some testing has validated that everything (including the actual physics simulations) can be successfully reproduced on hardware such as the Raspberry Pi 4 running a 64-bit operating system, which is an `ARM64` device.


Using Rosetta Emulation, the docker image for the physics simulation was able to run on a first-generation M1 Macbook Mini, and all other results were reproduced using `ARM64` builds of Python 3.9.6.
That is more or less the the extent of testing that has been conducted as of December 2021.
`mud_run_all` has been tested on several `M1` MacBooks (also `ARM64`), and it is the author's intent to eventually build an ARM64 docker image capable of running the physics simulations, so despite the `LaTeX` images supporting both platforms, the `python` images have not been built as native multi-arch images at the time of writing.
Please open an issue if you would like help compiling or reproducing results on such machines. It is possible, just less convenient.

Docker buildx is being used with Github Actions to publish multi-arch images for the LaTeX dependencies.
Eventually the plan is to have three images:
- One for data-generation
- One for demonstration of work
- One for compilation of PDF / Presentation


Note:
Most examples/figures-related testing on the M1 occurred using native Python (no emulation), and the following would have served as helpful information:

For the M1, make sure brew / python are running as `ARM64`, (you can check using `file $(which python)` and `file $(which brew)`), and then install dependencies in this order to avoid unecessarily building major scientific python libraries from source.

```sh
brew install scipy numpy
pip3 install mud-examples
```


[mamba-site]: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html
[mamba-hub]: https://hub.docker.com/r/mambaorg/micromamba/tags?page=1&name=bullseye
[fenics-hub]: https://hub.docker.com/r/mindthemath/fenics/tags
[python-hub]: https://hub.docker.com/_/python/
[python-thesis-hub]: https://hub.docker.com/r/mathematicalmichael/python-thesis/tags
[latex-thesis-hub]: https://hub.docker.com/r/mathematicalmichael/latex-thesis/tags
