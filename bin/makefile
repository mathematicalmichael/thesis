# check for required binaries, fail gracefully with helpful error message.
REQUIRED_BINS := docker
$(foreach bin,$(REQUIRED_BINS),\
    $(if $(shell command -v $(bin) 2> /dev/null),$(info Found `$(bin)`),$(error Please install `$(bin)`)))

# targets that are labeled as PHONY are treated as always needing an update
# a file doesn't actually need to exist for it to run
.PHONY: all full_image latex_image python_image

# the first real target is the one used when no other arguments are passed to `make`
# by creating a dependency on the pdf, we trigger a compilation by default.
all: latex_image python_image

# bare-bones dependencies to build image
latex_image: Dockerfile
	docker build -t latex:minimal -f Dockerfile .
	docker tag latex:minimal latex:latest

# extras to build posters/graphics
full_image: Dockerfile-full latex_image
	docker build -t latex:full -f Dockerfile-full .

python_image: Dockerfile-python
	docker build -t python:thesis -f Dockerfile-python .
