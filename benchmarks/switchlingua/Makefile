.PHONY: install

ifeq ($(OS),Windows_NT)
VENV_PYTHON ?= ..\\..\\.venv\\Scripts\\python.exe
else
VENV_PYTHON ?= ../../.venv/bin/python
endif

BENCHMARK_EXTRAS ?= qwen,mms_lid
OUTDIR ?= ext/switchlingua
MANIFEST ?= $(OUTDIR)/manifest_bootstrap.jsonl

install:
	$(VENV_PYTHON) -m pip install uv
	uv pip install -e "../..[$(BENCHMARK_EXTRAS)]" -r requirements.txt
	$(VENV_PYTHON) ../../tools/switchlingua_download.py --audio-only --outdir ../../$(OUTDIR) --manifest ../../$(MANIFEST) $(if $(ALLOW_PATTERN),--allow-pattern "$(ALLOW_PATTERN)",)
