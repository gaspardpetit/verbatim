# Build Instructions

## Supported Python Versions
This project supports **Python 3.10**, **3.11**, and **3.12**. Ensure you have one of these versions installed on your system before proceeding.

## Setting Up the Environment

### 1. Create a Virtual Environment
Using a virtual environment is recommended to isolate project dependencies and avoid conflicts with other Python projects.

#### On Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

#### On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Update `pip`
Before installing the project dependencies, ensure that `pip` is up to date:
```bash
pip install --upgrade pip
```

### 3. Install Dependencies

#### Basic Installation:
Install the required dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

#### GPU Support:
- **macOS (darwin):**
  ```bash
  pip install -r requirements-gpu.txt
  ```

- **Windows with CUDA Support:**
  If you are using GPU acceleration on Windows, you'll need to specify the appropriate CUDA version. For example, for CUDA 12.4:
  ```bash
  pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-gpu.txt
  ```
  Adjust the CUDA version (`cu124`) in the URL as necessary for your setup.

## Using Astral UV for Advanced Dependency Management
The project uses **Astral UV** for advanced dependency management, including custom indices and conditional dependencies.

### Install Astral UV
Astral UV can be installed using pip:
```bash
pip install uv
```

### Install Dependencies with Astral UV
Use the `uv` CLI to install dependencies as per the configurations in `pyproject.toml`:
```bash
uv pip install .  --group dev
# or if you will be working on the project:
# uv pip install -e .  --group dev
# or if you also want to pull pyannote diarization support:
# uv pip install -e ".[diarization]" --group dev
```
This will resolve dependencies, including those with custom indices or conditional markers.

## Using the `pyproject.toml`
This project uses a `pyproject.toml` file for dependency and build configuration. To interact with this framework, you can use tools such as `pip`, `poetry`, or Astral UV:

### Install Dependencies with `pip`
If not already done, dependencies listed in `pyproject.toml` can be installed with:
```bash
pip install .
```

### Using `Poetry` (Optional)
If you prefer using `poetry` for dependency management and builds:
1. Install Poetry:
   ```bash
   pip install poetry
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the Poetry shell (optional):
   ```bash
   poetry shell
   ```

## Additional Notes
- The project uses the `astral-uv` library for  dependency management. Ensure it is properly installed as part of the dependencies.
- For advanced builds or custom configurations, refer to the comments in `pyproject.toml` and the respective tool documentation.

## Verifying the Installation
To verify that everything is set up correctly:
1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. Run the project's test suite or a basic command to confirm:
   ```bash
   python -m unittest discover
   ```
   or
   ```bash
   python run.py
   ```

You should now be ready to develop and run the project

# Upgrading dependencies

```
pip install uv --upgrade
pip install --upgrade pip
uv lock --upgrade
```

# Regenerating the requirements.txt files

```
uv export --no-hashes --extra diarization > .\requirements.txt
uv export --no-hashes --extra diarization --extra cuda_gpu > .\requirements-gpu.txt
```

On windows, make sure the output is in utf-8, not utf-16 is it would be by default:

```{powershell}
uv export --no-hashes --extra diarization | Out-File -FilePath .\requirements.txt -Encoding utf8
uv export --no-hashes --extra diarization --extra cuda_gpu | Out-File -FilePath .\requirements-gpu.txt -Encoding utf8
```

You can add `--no-dev` to `uv export` to ignore the dev dependencies.


# Before submitting a change request

```
# pip install ruff
ruff check verbatim tests # try `ruff check --fix verbatim tests`if you see some failures
# pip install pylint
pylint verbatim $(git ls-files 'tests/*.py')
# pip install flake8
flake8 --count --select=E9,F63,F7,F82 --show-source --statistics verbatim tests
flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics verbatim tests
# pip install bandit
bandit -r verbatim tests
```
