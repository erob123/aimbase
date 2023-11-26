## ************ ENV VAR INIT BEFORE IMPORTS ************ ##
# Make sure to set ENVIRONMENT, ENV_VAR_FOLDER, and SECRETS in your environment,
# outside of any .env file.  This is to ensure that the correct environment
# variables are loaded before the app is initialized.
# Default values are: ENVIRONMENT=local, ENV_VAR_FOLDER=./env_vars, SECRETS=False if not set here
import os

os.environ["ENVIRONMENT"] = "local"
os.environ["ENV_VAR_FOLDER"] = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "env_vars"
)
os.environ["SECRETS"] = "False"
## ************ ENV VAR INIT BEFORE IMPORTS ************ ##

from aimbase.crud.base import CRUDBaseAIModel
from aimbase.db.base import BaseAIModel
from aimbase.initializer import AimbaseInitializer
from aimbase.routers.sentence_transformers_router import (
    SentenceTransformersRouter,
)
from aimbase.dependencies import get_minio
from aimbase.crud.sentence_transformers_vector import (
    CRUDSentenceTransformersVectorStore,
)
from aimbase.db.vector import AllMiniVectorStore
from instarest import (
    AppBase,
    DeclarativeBase,
    SchemaBase,
    Initializer,
    get_db,
)

from aimbase.services.sentence_transformers_inference import (
    SentenceTransformersInferenceService,
)

# TODO: import to __init__.py for aimbase and update imports here
Initializer(DeclarativeBase).execute()
AimbaseInitializer().execute()

# built pydantic data transfer schemas automagically
crud_ai_schemas = SchemaBase(BaseAIModel)
crud_vector_schemas = SchemaBase(AllMiniVectorStore)

# build db services automagically
crud_ai_test = CRUDBaseAIModel(BaseAIModel)
crud_vector_test = CRUDSentenceTransformersVectorStore(AllMiniVectorStore)

## ************ DEV INITIALIZATION ONLY (if desired to simulate
#  no internet connection...will auto init on first endpoint hit, but
#  will not auto-upload to minio) ************ ##
SentenceTransformersInferenceService(
    model_name="all-MiniLM-L6-v2",
    db=next(get_db()),
    crud=crud_ai_test,
    s3=get_minio(),
    prioritize_internet_download=False,
).dev_init()
## ************ DEV INITIALIZATION ONLY ************ ##

# build ai router automagically
test_router = SentenceTransformersRouter(
    model_name="all-MiniLM-L6-v2",
    schema_base=crud_vector_schemas,
    crud_ai_base=crud_ai_test,
    crud_base=crud_vector_test,
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
