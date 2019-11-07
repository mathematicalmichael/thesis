CHAPTERS = $(shell find . -type f -name 'chapter*.tex')
APPENDIX = $(shell find . -type f -name 'appendix*.tex')
IMAGES = $(shell find . -type f -name '*.png')
REFS = $(shell find . -type f -name 'references*.bib')

all: dissertation.pdf clean

dissertation.pdf: dissertation.tex *.tex $(CHAPTERS) $(APPENDIX) $(REFS) $(IMAGES) ucdenver-dissertation.cls
	latexmk -gg -pdf -bibtex dissertation.tex

clean:
	latexmk -c dissertation.tex
	/bin/rm -f *.spl
	/bin/rm -f *.bbl
