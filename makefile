# check for required binaries, fail gracefully with helpful error message.
REQUIRED_BINS := latexmk python
$(foreach bin,$(REQUIRED_BINS),\
    $(if $(shell command -v $(bin) 2> /dev/null),$(info Found `$(bin)`),$(error Please install `$(bin)`)))

# find all files required to compile / files that should trigger an update
CHAPTERS = $(shell find . -type f -name 'chapter*.tex')
APPENDIX = $(shell find . -type f -name 'appendix*.tex')

# changes to images/figures: add/edit this line to reflect your file types
# if you create a new variable (perhaps to track a folder), add it to target 
IMAGES = $(shell find . -type f -name '*.png') 
FIGURES = $(shell find . -type f -name '*.pdf' | grep 'figures')

REFS = $(shell find . -type f -name 'references*.bib')
# custom latex environment/styles for python, bash, etc. (syntax highlighting)
ENVS = $(shell find . -type f -name '*env.tex') 

# file name (without .tex)
FILENAME = dissertation

# dependency list: if changes detected in dependency, rebuild target
TEXFILES := \
	$(FILENAME) \
	abstract \
	notation \
	env/newcommands \
	env/usepackages \

# adds .tex to path name
TEXS = $(patsubst %, %.tex, $(TEXFILES))

# style-file dependendencies (unlikely to change these, but just in case)
DEPS := \
	ref/ucdDissertation.bst \
	ucdenver-dissertation.cls \

# targets that are labeled as PHONY are treated as always needing an update
# a file doesn't actually need to exist for it to run
.PHONY: all clean examples

# the first real target is the one used when no other arguments are passed to `make`
# by creating a dependency on the pdf, we trigger a compilation by default.
all: $(FILENAME).pdf

# our main target
$(FILENAME).pdf: $(TEXS) $(CHAPTERS) $(APPENDIX) $(REFS) $(IMAGES) $(FIGURES) $(ENVS) $(DEPS)
	latexmk -gg -pdf -bibtex $(FILENAME).tex

examples:
	cd examples && \
	bash examples_linear.sh && \
	bash heatrod_example.sh
	echo "All examples built."

clean:
	latexmk -c $(FILENAME).tex
	/bin/rm -f *.spl
	/bin/rm -f *.bbl
