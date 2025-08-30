.PHONY: check lint type sec

# Run local static checks. Ensure tools are installed: pip install ruff pylint flake8 bandit pyright
check: lint type sec

lint:
	ruff check verbatim tests
	flake8 verbatim tests
	pylint --disable=import-error verbatim $$(git ls-files 'tests/*.py')

type:
	pyright

sec:
	bandit -r verbatim tests run.py

# Format code using Ruff formatter
.PHONY: fmt
fmt:
	ruff format .

# Run test suite (quick)
.PHONY: test
test:
	CUDA_VISIBLE_DEVICES=-1 pytest -q

# Mirror CI fast matrix locally: checks + fast tests
.PHONY: ci
ci: check
	CUDA_VISIBLE_DEVICES=-1 pytest -q -k "not test_diarization_metrics_long and not SaTSentenceTokenizer"

