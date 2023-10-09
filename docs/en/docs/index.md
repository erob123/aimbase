<!-- adapted from FastAPI's index.md at https://github.com/tiangolo/fastapi -->
<p align="center">
    <em>Declarative, instant REST APIs for base AI Models based on instarest, a FastAPI, Pydantic, SQLAlchemy, and PostgreSQL library, and MinIO.</em>
</p>

---

**Documentation**: <a href="https://aimbase.erob.io/" class="external-link" target="_blank">aimbase.erob.io</a>

**Source Code**: <a href="https://github.com/erob123/aimbase" class="external-link" target="_blank">github.com/erob123/aimbase</a>

---

`Aimbase` offers an opinionated yet flexible abstraction, seamlessly integrating with large object storage to safeguard proprietary or data-controlled model weights. Leveraging the instantaneous database and API configuration capabilities provided by `Instarest`, `Aimbase` presents a declarative, extensible, and opinionated framework tailored for crafting REST APIs dedicated to AI Models. It streamlines development by eliminating redundant boilerplate and promoting code reusability, proving especially valuable when working with models associated with sensitive or proprietary data.

**_Our goal is to help you turn months of work into days and thousands of lines of code into less than a hundred._**

By using `Aimbase`, you will notice:

- **Simplicity**: your typical multi-folder, multi-file, multi-class, multi-method, multi-line, multi-annotation, multi-configuration FastAPI application will be reduced to a single file with a few lines of code.
- **Consistency**: your application will be built on a consistent, declarative, and opinionated foundation, making it easier to understand and maintain.
- **Speed**: your application will be built on a foundation that is designed to be fast, both in terms of development and runtime performance.
- **Fewer Unit Tests**: your application will be built on a foundation that is designed to be correct, reducing the need for extensive unit testing. Complete code coverage can be achieved with a handful of unit tests.
- **Easy**: Designed to be easy to use and learn. Less time reading docs.
- **Extensible**: `Aimbase` is built to be modular, and as such is easy to extend so that you can have your own custom, realizable opinions and abstractions. Frameworks such as **YourFramework** and **Chainbase** are built on top of `Aimbase` in this way.
- **Standards-based**: Based on FastAPI, which itself is based on (and fully compatible with) the open standards for APIs: <a href="https://github.com/OAI/OpenAPI-Specification" class="external-link" target="_blank">OpenAPI</a> (previously known as Swagger) and <a href="https://json-schema.org/" class="external-link" target="_blank">JSON Schema</a>

## Requirements

- Python 3.11+
- A PostgreSQL database (use `docker-compose` to get one up and running quickly)
- MinIO (used for large-object storage of proprietary or data-controlled model weights and configurations)

## Installation

<div class="termy">

```console
$ pip install aimbase

---> 100%
```

</div>

## Example

### Create it

Let's create a comprehensive, database-backed, large-object storage-enabled, type-checked,
versioned REST API tailored for handling proprietary or data-controlled models with **Aimbase** in just five minutes:

* Create a file `main.py` with:

```Python
from aimbase.crud.base import CRUDBaseAIModel
from aimbase.db.base import BaseAIModel, FineTunedAIModel, FineTunedAIModelWithBaseModel
from aimbase.initializer import AimbaseInitializer
from aimbase.routers.sentence_transformer_router import SentenceTransformersRouter
from aimbase.dependencies import get_minio
from instarest import (
    AppBase,
    DeclarativeBase,
    SchemaBase,
    Initializer,
    get_db,
)

from aimbase.services.sentence_transformer_inference import (
    SentenceTransformerInferenceService,
)

Initializer(DeclarativeBase).execute()
AimbaseInitializer().execute()

# built pydantic data transfer schemas automagically
crud_schemas = SchemaBase(BaseAIModel)

# build db service automagically
crud_test = CRUDBaseAIModel(BaseAIModel)

## ************ DEV INITIALIZATION ONLY (if desired to simulate
#  no internet connection...will auto init on first endpoint hit, but
#  will not auto-upload to minio) ************ ##
SentenceTransformerInferenceService(
    model_name="all-MiniLM-L6-v2",
    db=next(get_db()),
    crud=crud_test,
    s3=get_minio(),
    prioritize_internet_download=False,
).dev_init()
## ************ DEV INITIALIZATION ONLY ************ ##

# build ai router automagically
test_router = SentenceTransformersRouter(
    model_name="all-MiniLM-L6-v2",
    schema_base=crud_schemas,
    crud_base=crud_test,
    prefix="/sentences",
    allow_delete=True,
)

# setup base up from routers
app_base = AppBase(
    crud_routers=[test_router], app_name="Aimbase Inference Test App API"
)

# automagic and version app
auto_app = app_base.get_autowired_app()

# core underlying app
app = app_base.get_core_app()
```

### Setup the database and MinIO

If you already have a PostgreSQL database and MinIO running, you can skip this step.

If not, we will launch them via local containers:

1. First, make sure that you have docker installed and running. If you don't, <a href="https://docs.docker.com/get-docker/" class="external-link" target="_blank">you can install it by following the directions at this link</a>.

1. Download the `aimbase` docker-compose file:

```console
$ curl -O https://raw.githubusercontent.com/erob123/aimbase/main/docker-compose.yml
```

1. Launch the database and MinIO via docker-compose from the same directory as the `docker-compose.yml` file:

```console
$ docker-compose up --build
```

### Tell your app how to connect to the database and MinIO

Aimbase is set up to automatically connect to PostgreSQL and MinIO with the following environment variables. To get started, create a file named `local.env` in the same directory as `main.py` with the following contents:

```console
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=postgres

DOCS_UI_ROOT_PATH=""
LOG_LEVEL="DEBUG"

MINIO_BUCKET_NAME=test
MINIO_ACCESS_KEY=miniouser
MINIO_SECRET_KEY=minioadmin
MINIO_ENDPOINT_URL=localhost:9000
MINIO_REGION=""
MINIO_SECURE=False
```

This will allow your app to connect to the database nad MinIO large
object storage on launch. For reference, these values are defined
within the `docker-compose.yml` file for local development, but for
production they will come from your password defined through your database provider.

### Run it

You should now have two files in the same directory: `main.py` and `local.env`. Let's run the app with:

<div class="termy">

```console
$ uvicorn main:auto_app --reload

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [28720]
INFO:     Started server process [28722]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

</div>

<details markdown="1">
<summary>About the command <code>uvicorn main:app --reload</code>...</summary>

The command `uvicorn main:auto_app` refers to:

- `main`: the file `main.py` (the Python "module").
- `auto_app`: the object created inside of `main.py` with the line `auto_app = app_base.get_autowired_app()`.
- `--reload`: make the server restart after code changes. Only do this for development.

</details>

### Interactive API docs

Now go to <a href="http://127.0.0.1:8000/v1/docs" class="external-link" target="_blank">http://127.0.0.1:8000/v1/docs</a>.

You will see the automatic interactive API documentation (provided by <a href="https://github.com/swagger-api/swagger-ui" class="external-link" target="_blank">Swagger UI</a>):

Congratulations! You have just created your first fully-functional REST API with Aimbase, implementing AI-based sentence embeddings backed by a database in just a few lines.

### Check it

#### Send a test sentence to the API

In the interactive docs (or using `curl` if you prefer), go to the POST operation for `/sentences/encode` and try it. Send this JSON body:

```JSON
{
  "documents": ["This is a test sentence."]
}
```

You should see the `curl` that was sent

```console
$ curl -X 'POST' \
'http://localhost:8000/v1/sentences/encode' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{"documents": ["This is a test sentence."]}'
```

and the response received (with calculated `embeddings`):

```JSON
{
  "embeddings": [[0.123, 0.456, 0.789, ...]]
}
```

And that's it! You have successfully created a base API for `sentence-transformers` embeddings. Continue through the documentation to see how to build more extensible APIs using `sentence-transformers` and other AI models
with `aimbase`.

<p align="center">
    <em>All done in under 50 lines of code.</em>
</p>

### Alternative API docs

If you prefer, go to <a href="http://127.0.0.1:8000/v1/redoc" class="external-link" target="_blank">http://127.0.0.1:8000/v1/redoc</a>.

You will see the alternative automatic documentation (provided by <a href="https://github.com/Rebilly/ReDoc" class="external-link" target="_blank">ReDoc</a>):

## License

This project is licensed under the terms of the MIT license.
