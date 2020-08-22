# check for required binaries, fail gracefully with helpful error message.
REQUIRED_BINS := docker
$(foreach bin,$(REQUIRED_BINS),\
    $(if $(shell command -v $(bin) 2> /dev/null),$(info Found `$(bin)`),$(error Please install `$(bin)`)))

# targets that are labeled as PHONY are treated as always needing an update
# a file doesn't actually need to exist for it to run
.PHONY: all clean \
	latex_image \
	python_image fenics_image conda_image \
	publish path

# the first real target is the one used when no other arguments are passed to `make`
# by creating a dependency on the pdf, we trigger a compilation by default.
all: latex_image python_image

# bare-bones dependencies to build image
latex_image: Dockerfile
	docker build -t latex:thesis -f Dockerfile .
	docker tag latex:thesis latex:latest


# TODO change to Dockerfile-fenics and make the python one based on python3
# in order to do this, you need to fully separate out the generation of data
# for examples and the analysis of that data.
python_image: fenics_image

fenics_image: Dockerfile-python
	docker build -t python:thesis -f Dockerfile-python .

conda_image: Dockerfile-conda
		docker build -t python:thesis -f Dockerfile-conda .

publish: python_image
	docker tag python:thesis mathematicalmichael/python:thesis
	docker push mathematicalmichael/python:thesis

# is this working??
path:
	@export PATH=$(shell pwd):${PATH}

clean:
	docker system prune
