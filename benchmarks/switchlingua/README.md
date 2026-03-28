# SwitchLingua Benchmark

This folder is the benchmark bootstrap corner for SwitchLingua. For now it carries the local installation/bootstrap scaffolding plus the dataset downloader and master-manifest builder so the first PRs can stay small. The rest of the benchmark scripts still live under `tools/`.

## Layout

- `Makefile`: benchmark-local setup and run entrypoints
- `requirements.txt`: extra helper dependency needed by the downloader
- `scripts/download.py`: benchmark-local dataset bootstrap script
- `scripts/manifest.py`: benchmark-local master-manifest builder

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

## Current Runner Location

The benchmark scripts currently stay outside this folder:
- `tools/switchlingua_manifest.py`
- `tools/switchlingua_benchmark.py`
- `tools/switchlingua_run_all.py`
- `tools/switchlingua_report.py`

The runner/report scripts still stay outside this folder for now.
