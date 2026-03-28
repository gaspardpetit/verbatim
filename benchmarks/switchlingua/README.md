# SwitchLingua Benchmark

This folder is the benchmark corner for SwitchLingua. It carries local installation/bootstrap scaffolding, the dataset downloader, the master-manifest builder, and benchmark-local runner/config files so the workflow can move out of `tools/` in small reviewable steps.

## Layout

- `Makefile`: benchmark-local setup and run entrypoints
- `requirements.txt`: extra helper dependency needed by the downloader
- `scripts/download.py`: benchmark-local dataset bootstrap script
- `scripts/manifest.py`: benchmark-local master-manifest builder
- `scripts/benchmark.py`: benchmark-local runner
- `scripts/run_all.py`: benchmark-local sequential launcher
- `scripts/systems.py`: benchmark config loader and validator
- `systems.yaml`: system definitions and argument overrides
- `benchmark.yaml`: enabled language/system set for `make benchmark`

## Install

Install the repo plus the benchmark helper dependency:

```bash
uv pip install -e ".[qwen,mms_lid]" -r benchmarks/switchlingua/requirements.txt
```

Add Voxtral support when needed:

```bash
uv pip install -e ".[qwen,mms_lid,voxtral]" -r benchmarks/switchlingua/requirements.txt
```

Or let the benchmark-local make target do it and start a dataset pull:

```bash
make -C benchmarks/switchlingua install
```

Defaults for `make install`:
- installs `.[qwen,mms_lid]`
- installs `benchmarks/switchlingua/requirements.txt`
- uses `MAX_WORKERS=1` for a conservative Hugging Face sync
- starts syncing the full audio dataset into `ext/switchlingua`
- writes a benchmark-owned master manifest to `benchmarks/switchlingua/manifests/manifest_bootstrap.jsonl`

Override that on demand:

```bash
make -C benchmarks/switchlingua install BENCHMARK_EXTRAS="qwen,mms_lid,voxtral" ALLOW_PATTERN="Arabic/*.m4a"
```

Set `ALLOW_PATTERN` only when you want to restrict the sync to a subset.
Increase concurrency only if your HF quota can sustain it:

```bash
make -C benchmarks/switchlingua install MAX_WORKERS=4
```

If Hugging Face rate limits the download (HTTP 429), the downloader sleeps for 60 seconds and retries automatically. Override with:
- `SWITCHLINGUA_RATE_LIMIT_SLEEP_SECONDS`
- `SWITCHLINGUA_RATE_LIMIT_MAX_RETRIES`

The manifest generator can use either a `metadata.*` file or per-language metadata files like `Arabic.csv` located at the dataset root.

The dataset is gated on Hugging Face. Accept the terms and set `HUGGINGFACE_TOKEN` or `HF_TOKEN` before downloading.

The bootstrap manifest is normalized for direct benchmark consumption:
- `languages` is normalized and filled with dataset-level fallbacks when metadata is incomplete
- `text` is reduced to the per-audio utterance instead of preserving a raw multi-turn payload
- `normalized` is precomputed for scoring

## Run

List benchmark systems from the local config:

```bash
python benchmarks/switchlingua/scripts/benchmark.py --list-systems
```

Run one language/system slice:

```bash
python benchmarks/switchlingua/scripts/benchmark.py --lang french --systems qwen_mms
```

Run the configured default matrix:

```bash
make -C benchmarks/switchlingua benchmark
```

The runner config is split in two:
- `systems.yaml` defines each named system and its overrides
- `benchmark.yaml` defines the default language/system set used by `make benchmark`

By default, benchmark outputs now land under `benchmarks/switchlingua/out/`.
