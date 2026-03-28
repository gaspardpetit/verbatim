.PHONY: install benchmark

ifeq ($(OS),Windows_NT)
VENV_PYTHON ?= ..\\..\\.venv\\Scripts\\python.exe
UV_PIP_INSTALL = cmd /C "set UV_LINK_MODE=$(UV_LINK_MODE)&& uv pip install -e ../..[$(BENCHMARK_EXTRAS)] -r requirements.txt"
else
VENV_PYTHON ?= ../../.venv/bin/python
UV_PIP_INSTALL = UV_LINK_MODE=$(UV_LINK_MODE) uv pip install -e "../..[$(BENCHMARK_EXTRAS)]" -r requirements.txt
endif

UV_LINK_MODE ?= copy
BENCHMARK_EXTRAS ?= qwen,mms_lid
MAX_WORKERS ?= 1
OUTDIR ?= ext/switchlingua
MANIFEST ?= benchmarks/switchlingua/manifests/manifest_bootstrap.jsonl
SYSTEMS_CONFIG ?= benchmarks/switchlingua/systems.yaml
BENCHMARK_CONFIG ?= benchmarks/switchlingua/benchmark.yaml

install:
	$(VENV_PYTHON) -m pip install uv
	$(UV_PIP_INSTALL)
	$(VENV_PYTHON) scripts/download.py --max-workers $(MAX_WORKERS) --audio-only --outdir ../../$(OUTDIR) $(if $(ALLOW_PATTERN),--allow-pattern "$(ALLOW_PATTERN)",)
	$(VENV_PYTHON) scripts/manifest.py --audio-root ../../$(OUTDIR)/SwitchLingua_audio --manifest ../../$(MANIFEST) $(if $(ALLOW_PATTERN),--allow-pattern "$(ALLOW_PATTERN)",)

benchmark:
	$(VENV_PYTHON) scripts/run_all.py --benchmark-config ../../$(BENCHMARK_CONFIG) --systems-config ../../$(SYSTEMS_CONFIG)
