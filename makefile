CHAPTERS = $(shell find . -type f -name 'chapter*.tex')
APPENDIX = $(shell find . -type f -name 'appendix*.tex')
REFS = $(shell find . -type f -name 'references*.bib')

.PHONY: all clean

all: dissertation.pdf dissertation-coadvisors.pdf

dissertation.pdf: dissertation.tex *.tex $(CHAPTERS) $(APPENDIX) $(REFS) newcommands.tex outline.tex ucdenver-dissertation.cls
	latexmk -gg -pdf -bibtex dissertation.tex

dissertation-coadvisors.pdf: dissertation-coadvisors.tex $(CHAPTERS) $(APPENDIX) newcommands.tex ucdenver-dissertation-coadvisors.cls
	latexmk -gg -pdf -bibtex dissertation-coadvisors.tex

clean:
	latexmk -c dissertation.tex 
	/bin/rm -f *.spl
	/bin/rm -f *.bbl
