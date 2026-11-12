PYTHON := .venv/bin/python

.PHONY: setup check smoke pilot full report report-watch assignment2 recent-baselines recent-status independent-response independent-status
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
assignment2:
	$(PYTHON) scripts/build_assignment2_report.py --group-id 11 --package $(ARGS)
recent-baselines:
	$(PYTHON) scripts/launch_recent_baselines.py launch $(ARGS)
recent-status:
	$(PYTHON) scripts/launch_recent_baselines.py status
independent-response:
	$(PYTHON) scripts/launch_independent_response.py launch
independent-status:
	$(PYTHON) scripts/launch_independent_response.py status
