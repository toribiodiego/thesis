.PHONY: all clean

all: slides.pdf

slides.pdf: slides.tex
	pdflatex slides
	pdflatex slides

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.synctex.gz *.vrb
