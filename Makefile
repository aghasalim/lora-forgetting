.PHONY: setup data baseline train eval forgetting report app test docker clean report-check
PY := .venv/bin/python

setup:
	python3.12 -m venv .venv && .venv/bin/pip install -U pip \
	  && .venv/bin/pip install -r requirements-train.txt

feasibility:      ## measure s/step and memory before committing to a run
	$(PY) scripts/feasibility.py

data:             ## generate training data (asserts vendor disjointness)
	$(PY) -m src.loraft.data

baseline:         ## score the UN-tuned model -- run before training
	$(PY) -m src.loraft.evaluate

train:
	$(PY) -m src.loraft.train

eval:             ## score the fine-tuned model on the identical prompts
	$(PY) -m src.loraft.evaluate artifacts/adapter tuned

forgetting:       ## general-capability check, base and tuned
	$(PY) -m src.loraft.forgetting
	$(PY) -m src.loraft.forgetting artifacts/adapter tuned

report:
	$(PY) -m src.loraft.report

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest tests/ -q

docker:
	docker build -t lora-forgetting .

clean:
	rm -rf reports/*.jsonl reports/*.json artifacts/adapter

report-check:  ## fail if RESULTS.md no longer matches the generator
	$(PY) -m src.loraft.report --check
