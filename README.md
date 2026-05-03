# Portfolio

Flask portfolio app with a few machine-learning demo pages.

## Requirements

- Python 3.11
- pip

The bundled model pickle files were created with old scikit-learn versions, so
the app pins a Python 3.11-compatible scikit-learn stack and applies a small
compatibility shim while loading the models.

## Run Locally

```bash
pyenv shell 3.11.9
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/flask --app app run
```

Then open `http://127.0.0.1:5000`.

For the Procfile/gunicorn path:

```bash
.venv/bin/gunicorn app:app
```
