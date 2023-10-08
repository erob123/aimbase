<!-- adapted from FastAPI's index.md at https://github.com/tiangolo/fastapi -->
<p align="center">
    <em>Declarative, instant REST APIs based on FastAPI, Pydantic, SQLAlchemy, and PostgreSQL</em>
</p>

---

**Documentation**: <a href="https://erob123.github.io/instarest" target="_blank">https://erob123.github.io/instarest</a>

**Source Code**: <a href="https://github.com/erob123/instarest" target="_blank">https://github.com/erob123/instarest</a>

---

Instarest provides an opinionated, extensible implementation of a FastAPI, Pydantic, SQLAlchemy, and PostgreSQL stack. It is designed
to simplify and reuse common code patterns for these technologies, preventing excess boilerplate and duplicative code.

***Our goal is to help you turn months of work into days and thousands of lines of code into less than a hundred.***

By using instarest, you will notice

* **Simplicity**: your typical multi-folder, multi-file, multi-class, multi-method, multi-line, multi-annotation, multi-configuration
  FastAPI application will be reduced single file with a few lines of code.
* **Consistency**: your application will be built on a consistent, declarative, and opinionated foundation, making it easier to
    understand and maintain.
* **Speed**: your application will be built on a foundation that is designed to be fast, both in terms of development and runtime
    performance.
* **Fewer Unit Tests**: your application will be built on a foundation that is designed to be correct, reducing the need for
    extensive unit testing.  Complete code coverage can be achieved with a handful of unit tests.
* **Easy**: Designed to be easy to use and learn. Less time reading docs.
* **Extensible**: Instarest is built to be modular, and as such is easy to extend so that you can have your own custom, realizable opinionations
    and abstractions.  Frameworks such as **Aimbase** and **Chainbase** are built on top of Instarest in this way.
* **Standards-based**: Based on FastAPI, which itself is based on (and fully compatible with) the open standards for APIs: <a href="https://github.com/OAI/OpenAPI-Specification" class="external-link" target="_blank">OpenAPI</a> (previously known as Swagger) and <a href="https://json-schema.org/" class="external-link" target="_blank">JSON Schema</a>

## Requirements

* Python 3.11+
* A PostgreSQL database (use `docker-compose` to get one up and running quickly)

## Installation

<div class="termy">

```console
$ pip install instarest

---> 100%
```
</div>
## Example

### Create it

Let's create a complete database-backed, type-checked, versioned REST API with **Instarest** in five minutes:

* Create a file `main.py` with:

```Python
from instarest import (
    AppBase,
    DeclarativeBase,
    RESTRouter,
    SchemaBase,
    CRUDBase,
    Initializer,
)

from sqlalchemy import Column, String, Boolean


class EmptyTestModel(DeclarativeBase):
    bool_field = Column(Boolean(), default=False)
    title = Column(String(), default="title")

initializer.execute()

# built pydantic data transfer schemas automagically
crud_schemas = SchemaBase(EmptyTestModel)

# build crud db service automagically
crud_test = CRUDBase(EmptyTestModel)

# build crud router automagically
test_router = RESTRouter(
    schema_base=crud_schemas,
    crud_base=crud_test,
    prefix="/test",
    allow_delete=True,
)

# setup base up from routers
app_base = AppBase(crud_routers=[test_router], app_name="Test App API")

# automagic and version app
auto_app = app_base.get_autowired_app()

# core underlying app
app = app_base.get_core_app()
```

### Setup the database
If you already have a PostgreSQL database running, you can skip this step.

If not, we will launch via a local container:

1. First, make sure that you have docker installed and running.  If you don't, <a href="https://docs.docker.com/get-docker/" target="_blank">you can install it by following the directions at this link</a>.

1. Download the `instarest` docker-compose file:

```console
$ curl -O https://raw.githubusercontent.com/erob123/instarest/main/docker-compose.yml
```

1. Launch the database and pgadmin console via docker-compose from the same directory as the `docker-compose.yml` file:

```console
$ docker-compose up --build
```

### Tell your app how to connect to the database
Instarest is setup to automatically connect to PostgreSQL with just a few Environment variables.  To get started, create
a file named `local.env` in the same directory as `main.py` with the following contents:

```console
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=postgres
```

This will allow your app to connect to the database on launch.  For reference, these values are defined
within the `docker-compose.yml` file for local development, but for production they will come from your
password defined through your database provider.

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

* `main`: the file `main.py` (the Python "module").
* `auto_app`: the object created inside of `main.py` with the line `auto_app = app_base.get_autowired_app()`.
* `--reload`: make the server restart after code changes. Only do this for development.

</details>

### Interactive API docs

Now go to <a href="http://127.0.0.1:8000/v1/docs" class="external-link" target="_blank">http://127.0.0.1:8000/v1/docs</a>.

You will see the automatic interactive API documentation (provided by <a href="https://github.com/swagger-api/swagger-ui" class="external-link" target="_blank">Swagger UI</a>):

Congratulations! You have just created your first fully-functional REST API with Instarest, implementing all CRUD operations (Create, Read, Update, Delete).

### Check it

#### Step 1: Create a new item
In the interactive docs (or using `curl` if you prefer), go to the POST operation and try it. Send this JSON body:

```JSON
{"title": "Hello", "bool_field": true}
```

You should see the `curl` that was sent

```console
$ curl -X 'POST' \
'http://localhost:8000/v1/test/' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{"title": "Hello", "bool_field": true}'
```

and the response received (with unique `id`):

```JSON
{
  "bool_field": true,
  "title": "Hello",
  "id": "be79d6de-752d-468d-a649-2e0bc62fcf64"
}
```
#### Step 2: Read the item you just created
Open your browser at `http://127.0.0.1:8000/v1/test/{id}`, replacing `{id}` with the one you got in the previous step (for example, `be79d6de-752d-468d-a649-2e0bc62fcf64`).

You will see the JSON response as (with the same unique `id`):

```JSON
{"bool_field":true,"title":"Hello","id":"be79d6de-752d-468d-a649-2e0bc62fcf64"}
```

#### Step 3: Update the item you just created
In the interactive docs (or using `curl` if you prefer), go to the PUT operation and click `Try it out`. Put the same unique `id` into the
appropriate parameter box as pulled from the `GET` request in the previous step.  Then, send this JSON body:

```JSON
{"title": "Goodbye"}
```

You should see the `curl` that was sent

```console
$ curl -X 'PUT' \
'http://localhost:8000/v1/test/be79d6de-752d-468d-a649-2e0bc62fcf64' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{"title": "Goodbye"}'
```

and the response received (with the same unique `id`):

```JSON
{
  "bool_field": true,
  "title": "Goodbye",
  "id": "be79d6de-752d-468d-a649-2e0bc62fcf64"
}
```

Notice that the `title` field changed from `"Hello"` to `"Goodbye"`, but all other fields remained the same, even though we didn't send them.

#### Step 4: Read the item you just updated
Open your browser at `http://127.0.0.1:8000/v1/test/{id}`, replacing `{id}` with the same one you have been using.

You will see the JSON response as (with the same unique `id`):

```JSON
{"bool_field":true,"title":"Goodbye","id":"be79d6de-752d-468d-a649-2e0bc62fcf64"}
```

Notice that the `title` field changed from `"Hello"` to `"Goodbye"`, but all other fields remained the same.

#### Step 5: Delete the item you just created
In the interactive docs (or using `curl` if you prefer), go to the DELETE operation and click `Try it out`. Put the same unique `id` into the
appropriate parameter box that we have been using and click execute.

You should see the `curl` that was sent

```console
$ curl -X 'DELETE' \
'http://localhost:8000/v1/test/be79d6de-752d-468d-a649-2e0bc62fcf64' \
-H 'accept: application/json'
```

and the response received:

```JSON
{}
```

Now, without clearing the parameter box, click execute again.  You should see a 400 response
with the appropriate `id` in the error message:

```JSON
{
  "detail": "EmptyTest with id: be79d6de-752d-468d-a649-2e0bc62fcf64 not found"
}
```

A successful delete will return a 200 response with an empty JSON body.  A failed delete will return a 400 response with a JSON body containing
the error message.

#### Step 6: Read the item you just deleted
Open your browser at `http://127.0.0.1:8000/v1/test/{id}`, replacing `{id}` with the same one you have been using.

You will see a response with the appropriate `id` in the error message:

```JSON
{"detail":"EmptyTest with id: be79d6de-752d-468d-a649-2e0bc62fcf64 not found"}
```

And that's it!  You have a fully-functioning, single-store, persistent, versioned REST API with all CRUD operations implemented.

<p align="center">
    <em>All done in under 50 lines of code.</em>
</p>

### Alternative API docs

If you prefer, go to <a href="http://127.0.0.1:8000/v1/redoc" class="external-link" target="_blank">http://127.0.0.1:8000/v1/redoc</a>.

You will see the alternative automatic documentation (provided by <a href="https://github.com/Rebilly/ReDoc" class="external-link" target="_blank">ReDoc</a>):

## License

This project is licensed under the terms of the MIT license.
