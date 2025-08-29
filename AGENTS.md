# Repository Guidelines

## Project Structure & Module Organization
- `verbatim/`: Core library (audio, transcript, voices, eval, CLI in `main.py`).
- `tests/`: Unit tests (unittest-compatible, `test_*.py`, uses sample data in `tests/data/`).
- `doc/`: CLI and architecture docs; images in `doc/img/`.
- `samples/`: Example audio and reference outputs.
- Top-level helpers: `run.py` (local demo), `Dockerfile`, `docker-*.sh`, `BUILD.md`, `README.md`.

## Build, Test, and Development Commands
- Create env (Python 3.9–3.11): `python -m venv .venv && source .venv/bin/activate`.
- Install deps (preferred): `pip install uv && uv pip install .`.
- Alt install: `pip install -r requirements.txt` (GPU: `requirements-gpu.txt`).
- Run CLI: `verbatim samples/…/audio.mp3 -v -o out`.
- Local demo: `python run.py`.
- Tests (unittest): `python -m unittest -q` (or `pytest -q`).
- Lint/Sec checks: `ruff check verbatim tests`, `pylint verbatim $(git ls-files 'tests/*.py')`, `flake8 verbatim tests`, `bandit -r verbatim tests`.

## Architecture Overview
- Read `doc/architecture.md` for the end-to-end pipeline, data flow, and component contracts.
- Key areas: audio sources (`verbatim/audio/`), transcription/diarization/separation (`verbatim/voices/`), transcript formatting (`verbatim/transcript/`), metrics (`verbatim/eval/`).
- If you change stage interfaces or file formats, update `doc/architecture.md` and related diagrams in `doc/img/`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; line length: 150 (see `[tool.ruff]`).
- Follow PEP 8: `snake_case` for functions/vars, `CamelCase` for classes, modules lowercase with underscores.
- Prefer type hints and small, focused functions. Run `ruff` and `pylint` locally before PRs.

## Testing Guidelines
- Framework: `unittest` (pytest-compatible). Place tests under `tests/`, name `test_*.py`.
- Run a single test: `python -m unittest tests/test_audio.py -v`.
- Determinism: force CPU when needed `CUDA_VISIBLE_DEVICES=-1`.
- Include minimal sample inputs and assert concrete outputs (see `tests/test_pipeline.py`).

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope. Prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- PRs: clear description, rationale, and scope; link issues; include CLI output or artifact snippets when relevant; update docs (`doc/`, `README.md`). Ensure tests and linters pass.

## Security & Configuration Tips
- Diarization models require `HUGGINGFACE_TOKEN`. Provide via env or `.env` (e.g., `HUGGINGFACE_TOKEN=hf_***`). Do not commit secrets.
- For live mic on macOS/Linux, install PortAudio.
- Offline runs: prefer Docker with `--network none`. GPU users may need CUDA-specific PyTorch wheels (see `BUILD.md`).
