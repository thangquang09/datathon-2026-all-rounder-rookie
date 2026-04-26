# EDA v4 --- build pipeline (NeurIPS 2025 template)
#
# Targets:
#   make            : run pipeline + build main.pdf + verify (default)
#   make pipeline   : regenerate metrics_v4.json and all figures
#   make notebook   : execute notebook, refresh HTML export
#   make pdf        : compile main.pdf (assumes figures already exist)
#   make verify     : check main.pdf main content <= 4 pages
#                     (references + appendix + checklist do not count per NeurIPS)
#   make clean      : remove LaTeX build artifacts (keeps deliverables)
#   make distclean  : remove everything including figures and PDF

.PHONY: all pipeline notebook pdf verify pdf_vi verify_vi clean distclean

PY        := uv run python
JUPYTER   := uv run jupyter
LATEXMK   := latexmk -pdf -silent -interaction=nonstopmode
PDFLATEX  := pdflatex -interaction=nonstopmode -halt-on-error

all: pipeline pdf verify

pipeline:
	$(PY) eda_v4.py

notebook: pipeline
	$(JUPYTER) nbconvert --to notebook --execute eda_v4.ipynb --output eda_v4.ipynb
	$(JUPYTER) nbconvert --to html eda_v4.ipynb

pdf:
	$(LATEXMK) main.tex

verify: pdf
	@# NeurIPS template: main content must end on or before page 4.
	@# References, appendix, and checklist follow on later pages and do NOT count.
	@# We detect the last main-content page by locating where the first appendix
	@# (labelled app:audit) is registered in main.aux.
	@if [ ! -f main.aux ]; then \
		echo "[FAIL] main.aux missing; run 'make pdf' first."; exit 1; \
	fi

pdf_vi:
	$(LATEXMK) main_vietnam.tex

verify_vi: pdf_vi
	@if [ ! -f main_vietnam.aux ]; then \
		echo "[FAIL] main_vietnam.aux missing; run 'make pdf_vi' first."; exit 1; \
	fi
	@app_page=$$(grep -E '\\newlabel\{app:audit\}' main_vietnam.aux | sed -E 's/.*\{([0-9]+)\}\{.*/\1/' | head -1); \
	if [ -z "$$app_page" ]; then \
		echo "[FAIL] could not locate appendix start page (VI)."; exit 1; \
	fi; \
	main_pages=$$(($$app_page - 1)); \
	total_pages=$$(pdfinfo main_vietnam.pdf | awk '/Pages:/ {print $$2}'); \
	if [ "$$main_pages" -le 4 ]; then \
		echo "[OK] (VI) main content $$main_pages pages (<= 4); total $$total_pages pages"; \
		echo "     (references + appendix + checklist on pages $$app_page--$$total_pages)"; \
	else \
		echo "[FAIL] (VI) main content $$main_pages pages (> 4)."; exit 1; \
	fi
	@app_page=$$(grep -E '\\newlabel\{app:audit\}' main.aux | sed -E 's/.*\{([0-9]+)\}\{Data.*/\1/' | head -1); \
	if [ -z "$$app_page" ]; then \
		echo "[FAIL] could not locate appendix start page."; exit 1; \
	fi; \
	main_pages=$$(($$app_page - 1)); \
	total_pages=$$(pdfinfo main.pdf | awk '/Pages:/ {print $$2}'); \
	if [ "$$main_pages" -le 4 ]; then \
		echo "[OK] main content $$main_pages pages (<= 4); total $$total_pages pages"; \
		echo "     (references + appendix + checklist on pages $$app_page--$$total_pages)"; \
	else \
		echo "[FAIL] main content $$main_pages pages (> 4)."; exit 1; \
	fi

clean:
	rm -f main.aux main.log main.fls main.fdb_latexmk main.out main.synctex.gz \
	       main_vietnam.aux main_vietnam.log main_vietnam.fls main_vietnam.fdb_latexmk main_vietnam.out main_vietnam.synctex.gz \
	       main.build.log report.aux report.log report.fls report.fdb_latexmk \
	       report.out report.synctex.gz

distclean: clean
	rm -f main.pdf main_vietnam.pdf report.pdf metrics_v4.json eda_v4.html
	rm -rf images/*.pdf images/*.png
