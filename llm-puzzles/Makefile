VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

.PHONY: venv install test run api docker-build docker-run fmt

venv:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

install-local: venv
	$(PIP) install -r requirements.txt -r requirements-local.txt || true

test:
	$(VENV)/bin/pytest -q

run: install
	$(PY) run.py --task cyclic-coxeter-n16-r3 --splits fast --candidates baseline heuristic1

api: install
	$(VENV)/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t llm-api:latest .

docker-run:
	docker run --rm -e API_KEY=changeme -p 8000:8000 llm-api:latest
