.PHONY: install

ifeq ($(OS),Windows_NT)
VENV_PYTHON ?= ..\\..\\.venv\\Scripts\\python.exe
else
VENV_PYTHON ?= ../../.venv/bin/python
endif

UV_LINK_MODE ?= copy
BENCHMARK_EXTRAS ?= qwen,mms_lid
MAX_WORKERS ?= 1
OUTDIR ?= ext/switchlingua
MANIFEST ?= $(OUTDIR)/manifest_bootstrap.jsonl

install:
	$(VENV_PYTHON) -m pip install uv
	UV_LINK_MODE=$(UV_LINK_MODE) uv pip install -e "../..[$(BENCHMARK_EXTRAS)]" -r requirements.txt
	$(VENV_PYTHON) scripts/download.py --max-workers $(MAX_WORKERS) --audio-only --outdir ../../$(OUTDIR) --manifest ../../$(MANIFEST) $(if $(ALLOW_PATTERN),--allow-pattern "$(ALLOW_PATTERN)",)
