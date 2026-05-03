# Portfolio

Flask portfolio app with a few machine-learning demo pages.

## Requirements

- Python 3.12
- pip

The bundled model pickle files were created with old scikit-learn versions, so
the app pins a Python 3.12-compatible scikit-learn stack and applies a small
compatibility shim while loading the models.

## Run Locally

```bash
pyenv shell 3.12
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/flask --app app run
```

Then open `http://127.0.0.1:5000`.

## Vercel

Vercel currently supports Python 3.12, 3.13, and 3.14 for Python functions.
This repo tracks `.python-version` with `3.12` and pins dependencies that have
Python 3.12 wheels.

For the Procfile/gunicorn path:

```bash
.venv/bin/gunicorn app:app
```
