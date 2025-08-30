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

