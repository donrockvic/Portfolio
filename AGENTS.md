# Repository Guidelines

## Project Structure & Module Organization

This is a Flask portfolio app with several machine-learning demo routes.
`app.py` is the application entry point and contains route handlers, model
loading, CSRF setup, and small compatibility shims for older scikit-learn
pickles. HTML templates live in `templates/`, CSS and JavaScript assets live in
`static/css/` and `static/js/`, and images live in `static/images/`. Serialized
model artifacts are stored under `models/`, including loan-specific artifacts in
`models/loan/`. Deployment-related files are `Procfile`, `.python-version`, and
`requirements.txt`.

## Build, Test, and Development Commands

Use Python 3.12, matching `.python-version`.

```bash
pyenv shell 3.12
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/flask --app app run
```

The Flask command starts the local app at `http://127.0.0.1:5000`. For the
production-style entry point, run:

```bash
.venv/bin/gunicorn app:app
```

There is no dedicated build step; static assets are committed directly.

## Coding Style & Naming Conventions

Keep Python code simple and Flask-oriented. Use 4-space indentation, clear helper
functions, and route names that match the page or action they serve, such as
`/salary` and `/salary/output`. Prefer `snake_case` for new Python functions and
variables, even though some existing model variables use legacy mixed-case names.
Keep templates named by page purpose, for example `LoanPred.html` or
`SalaryPrediction.html`, and place new page assets under the appropriate
`static/` subdirectory.

## Testing Guidelines

No automated tests are currently present. Before opening a pull request, run the
Flask app locally and smoke-test the main routes: `/`, `/about`, `/projects`,
`/salary`, `/tweet`, and `/loan`. For changes to prediction forms, submit sample
form data and verify CSRF handling, model loading, and rendered prediction text.
If tests are added, prefer `pytest` and place them under `tests/` with names like
`test_routes.py` or `test_prediction_forms.py`.

## Commit & Pull Request Guidelines

Recent commits use short, direct subjects such as `Fix Vercel Python runtime
dependencies` and `updating resume`. Keep commits concise and imperative, for
example `Fix loan form validation`. Pull requests should include a brief summary,
manual test notes, linked issues when applicable, and screenshots for visible UI
changes. Call out dependency, Python runtime, or model pickle compatibility
changes explicitly.

## Security & Configuration Tips

Do not commit secrets, generated virtual environments, or local caches. Treat
pickle files as trusted artifacts only; avoid replacing them unless the matching
Python and scikit-learn versions are documented and manually verified.
