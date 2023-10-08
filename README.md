# Getting Started
To get started with `aimbase` for your application, visit the docs at <a href="https://aimbase.erob.io/" target="_blank">https://aimbase.erob.io/</a>

To contribute to `aimbase`, see the [contributing](#contributing) section below.

# Contributing
## Getting Started Locally
1. Launch postgres and pgadmin via docker-compose `docker-compose up --build`.

1. Keeping your containers running, open a new terminal with the root of this repo as the working directory. Install `poetry`: `pip install poetry` (or use `pipx` [on link here](https://python-poetry.org/docs/1.4#installing-with-pipx) if you prefer isolated envs, or consider using `conda`).

1. Create and enter the virtual environment: `poetry shell`

1. Install the dependencies `poetry install`

1. Start the app: `uvicorn examples.example_app:auto_app --reload`.

1. Open `localhost:8000/v1/docs` and start interacting with swagger!

1. You can shut down and your db / minio data will persist via docker volumes.

## MinIO
- The MinIO console is available at `localhost:9001`.  Login with user:`miniouser` and password:`minioadmin` if you launched the containers with `docker-compose up --build`.

## Hooks and Tests
1. Set up the precommit hook with `pre-commit install`.

1. Run tests and get coverage with `pytest --cov`, and get html reports for vs code live server (or any server) with `pytest --cov --cov-report=html:coverage_re`

1. Open a pull request against the repo!  Please write tests, your coverage will automatically be added as a comment to any PR via GH actions.

## Viewing Docs Locally
1. Install docs dependencies with `pip install -r requirements-docs.txt`.
1. Install `mkdocs` with `pip install mkdocs-material`.
1. Move into the `docs/en` directory via `cd docs/en`.
1. Run `mkdocs serve` to start a local server.
