# check for required binaries, fail gracefully with helpful error message.
REQUIRED_BINS := latexmk
$(foreach bin,$(REQUIRED_BINS),\
    $(if $(shell command -v $(bin) 2> /dev/null),$(info Found `$(bin)`),$(error Please install `$(bin)`)))

# find all files required to compile
CHAPTERS = $(shell find . -type f -name 'chapter*.tex')
APPENDIX = $(shell find . -type f -name 'appendix*.tex')
IMAGES = $(shell find . -type f -name '*.png')
REFS = $(shell find . -type f -name 'references*.bib')

# file name (without .tex)
FILENAME = dissertation

#.PHONY: all clean
all: $(FILENAME).pdf

$(FILENAME).pdf: $(FILENAME).tex *.tex $(CHAPTERS) $(APPENDIX) $(REFS) $(IMAGES) ucdenver-dissertation.cls
	latexmk -gg -pdf -bibtex $(FILENAME).tex

clean:
	latexmk -c $(FILENAME).tex
	/bin/rm -f *.spl
	/bin/rm -f *.bbl
