CHAPTERS = $(shell find . -type f -name 'chapter*.tex')
APPENDIX = $(shell find . -type f -name 'appendix*.tex')
REFS = $(shell find . -type f -name 'references*.bib')

.PHONY: all clean

all: dissertation.pdf

dissertation.pdf: dissertation.tex *.tex $(CHAPTERS) $(APPENDIX) $(REFS) ucdenver-dissertation.cls
	latexmk -gg -pdf -bibtex dissertation.tex

clean:
	latexmk -c dissertation.tex
	/bin/rm -f *.spl
	/bin/rm -f *.bbl
