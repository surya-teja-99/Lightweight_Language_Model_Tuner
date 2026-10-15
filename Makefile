PY := .venv/bin/python

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) scripts/train_demo.py

eval:
	$(PY) scripts/eval_demo.py

test:
	$(PY) -m pytest tests/ -q

demo: train eval test

clean:
	rm -rf results .venv __pycache__ lightweight_language_model_tuner/__pycache__ tests/__pycache__
