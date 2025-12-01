.PHONY: check lint type sec

# Run local static checks. Ensure tools are installed: pip install ruff pylint flake8 bandit pyright
check: lint type sec

lint:
	ruff check verbatim verbatim_batch verbatim_cli verbatim_diarization verbatim_rttm verbatim_serve verbatim_transcript tests
	flake8 verbatim verbatim_batch verbatim_cli verbatim_diarization verbatim_rttm verbatim_serve verbatim_transcript tests
	pylint --disable=import-error verbatim $$(git ls-files 'tests/*.py')

type:
	pyright
	
sec:
	bandit -r verbatim verbatim_batch verbatim_cli verbatim_diarization verbatim_rttm verbatim_serve verbatim_transcript tests run.py

# Format code using Ruff formatter
.PHONY: fmt
fmt:
	ruff format .

# Auto-fix with Ruff (format + quick fixes)
.PHONY: fix
fix:
	ruff format .
	ruff check --fix --select I .
	ruff check --fix verbatim verbatim_batch verbatim_cli verbatim_diarization verbatim_rttm verbatim_serve verbatim_transcript tests

# Run test suite (quick)
.PHONY: test
test:
	CUDA_VISIBLE_DEVICES=-1 pytest -q

# Mirror CI fast matrix locally: checks + fast tests
.PHONY: ci
ci: check
	CUDA_VISIBLE_DEVICES=-1 pytest -q -k "not test_diarization_metrics_long and not SaTSentenceTokenizer"

# Build and verify distribution locally (mirrors CI publish steps)
.PHONY: release
release:
	python -m pip install --upgrade pip build twine
	python -m build
	python -m twine check dist/*
	python -m venv .pkg-venv
	. .pkg-venv/bin/activate && \
	  python -m pip install --upgrade pip && \
	  python -m pip install dist/*.whl && \
	  python -c "import verbatim; print('Imported verbatim', getattr(verbatim, '__version__', 'unknown'))"

install:
	uv pip install .

.PHONY: docker docker-cpu docker-gpu

VERSION_PY := $(shell git describe --tags --always --dirty \
  | sed -E 's/^v//; s/-([0-9]+)-g[0-9a-f]+(-dirty)?$$/.post\1\2/; s/-dirty$$/.dev0/')

docker-cpu:
	docker build -t verbatim -f deploy/Dockerfile.cpu \
	  --build-arg VERSION=$(VERSION_PY) .

docker-gpu:
	docker build -t verbatim:gpu -f deploy/Dockerfile.gpu \
	  --build-arg VERSION=$(VERSION_PY) .