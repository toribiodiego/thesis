.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib chapters/*.tex
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz
	rm -f chapters/*.aux
