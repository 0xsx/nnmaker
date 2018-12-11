
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = nnetmaker
SOURCEDIR     = src
BUILDDIR      = .

all:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	mv html/* .
	mv html/.buildinfo .
	mv html/.nojekyll .
	rmdir html

clean:
	rm *.html
	rm objects.inv
	rm searchindex.js
	rm -Rf _sources
	rm -Rf _static
	rm -Rf doctrees
