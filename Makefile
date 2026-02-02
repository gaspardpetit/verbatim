.PHONY: check lint type sec install-cpu install-cuda

PYTHON ?= 3.11

ifeq ($(OS),Windows_NT)
PYTEST_CMD = cmd /C "set CUDA_VISIBLE_DEVICES=-1&& pytest -q"
PYTEST_CMD_CI = cmd /C "set CUDA_VISIBLE_DEVICES=-1&& pytest -q -k \"not test_diarization_metrics_long and not SaTSentenceTokenizer\""
else
PYTEST_CMD = CUDA_VISIBLE_DEVICES=-1 pytest -q
PYTEST_CMD_CI = CUDA_VISIBLE_DEVICES=-1 pytest -q -k "not test_diarization_metrics_long and not SaTSentenceTokenizer"
endif

# Run local static checks. Ensure tools are installed: pip install ruff pylint flake8 bandit pyright
check: lint type sec

lint:
	ruff check verbatim verbatim_audio verbatim_batch verbatim_cli verbatim_diarization verbatim_files verbatim_transcript tests
	flake8 verbatim verbatim_audio verbatim_batch verbatim_cli verbatim_diarization verbatim_files verbatim_transcript tests
	pylint --disable=import-error verbatim verbatim_audio verbatim_batch verbatim_cli verbatim_diarization verbatim_files verbatim_transcript $$(git ls-files 'tests/**/*.py')

type:
	pyright
	
sec:
	bandit -r verbatim verbatim_audio verbatim_batch verbatim_cli verbatim_diarization verbatim_files verbatim_transcript tests run.py

# Format code using Ruff formatter
.PHONY: fmt
fmt:
	ruff format .

# Auto-fix with Ruff (format + quick fixes)
.PHONY: fix
fix:
	ruff format .
	ruff check --fix --select I .
	ruff check --fix verbatim verbatim_audio verbatim_batch verbatim_cli verbatim_diarization verbatim_files verbatim_transcript tests

# Run test suite (quick)
.PHONY: test
test:
	$(PYTEST_CMD)

# Mirror CI fast matrix locally: checks + fast tests
.PHONY: ci
ci: check
	$(PYTEST_CMD_CI)

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

install-cpu:
	uv sync --python $(PYTHON) --extra diarization
	uv pip install --python $(PYTHON) torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu --reinstall
	@echo "Note: if diarization fails with torchcodec/FFmpeg errors, install FFmpeg 4–7 and set FFMPEG_DLL_DIR (Windows) or add ffmpeg to PATH."

install-cuda:
	uv sync --python $(PYTHON) --extra diarization
	uv pip install --python $(PYTHON) torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --reinstall
	@echo "Note: if diarization fails with torchcodec/FFmpeg errors, install FFmpeg 4–7 and set FFMPEG_DLL_DIR (Windows) or add ffmpeg to PATH."

.PHONY: docker docker-cpu docker-gpu

VERSION_PY = $(shell python -c "import re, subprocess, sys; \
try: \
  d=subprocess.check_output(['git','describe','--tags','--always','--dirty'], stderr=subprocess.DEVNULL).decode().strip(); \
except Exception: \
  print('0.0.0'); sys.exit(0); \
d=re.sub(r'^v','',d); \
d=re.sub(r'-([0-9]+)-g[0-9a-f]+(-dirty)?$$', r'.post\\1\\2', d); \
d=re.sub(r'-dirty$$', r'.dev0', d); \
print(d)") 

docker-cpu:
	docker build -t verbatim -f deploy/Dockerfile.cpu \
	  --build-arg VERSION=$(VERSION_PY) .

docker-gpu:
	docker build -t verbatim:gpu -f deploy/Dockerfile.gpu \
	  --build-arg VERSION=$(VERSION_PY) .
