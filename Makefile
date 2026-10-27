PYTHON := .venv/bin/python

.PHONY: setup check smoke pilot full report report-watch
setup:
	bash scripts/setup.sh
check:
	.venv/bin/ruff check src scripts tests
	$(PYTHON) -m pytest -q
smoke:
	bash scripts/run_experiments.sh --preset smoke
pilot:
	bash scripts/run_experiments.sh --preset pilot --resume $(ARGS)
full:
	bash scripts/run_experiments.sh --preset full --resume $(ARGS)
report:
	bash scripts/build_report.sh $(ARGS)
report-watch:
	$(PYTHON) scripts/build_active_report.py --background $(ARGS)
