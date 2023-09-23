from fastapi import Depends, HTTPException
from pydantic import BaseModel
from aimbase.services.sentence_transformer_inference import (
    SentenceTransformerInferenceService,
)
from aimbase.dependencies import get_minio
from instarest import RESTRouter
from instarest import get_db
from sqlalchemy.orm import Session
from minio import Minio


class SentenceTransformersRouter(RESTRouter):
    """
    FastAPI Router object wrapper for Sentence Transformers model.
    Uses pydantic BaseModel.

    **Parameters**

    Same as instarest parent router wrapper class, with the addition of:

    * `model_name`: Name of the Sentence Transformers model to use
    """

    model_name: str

    # override and do not call super() to prevent default CRUD endpoints
    def _add_endpoints(self):
        self._define_encode()

    def _define_encode(self):
        def build_model_not_initialized_error():
            return HTTPException(
                status_code=500,
                detail=f"{self.model_name} is not initialized",
            )

        class Embeddings(BaseModel):
            embeddings: list[list[float]] = []

        class Documents(BaseModel):
            documents: list[str] = []

        # ENCODE
        @self.router.post(
            "/encode",
            response_model=Embeddings,
            responses=self.responses,
            summary=f"Calculate embeddings for sentences or documents",
            response_description=f"Calculated embeddings",
        )
        async def encode(
            documents: Documents,
            db: Session = Depends(get_db),
            s3: Minio | None = Depends(get_minio),
        ) -> Embeddings:
            try:
                service = self._build_sentence_transformer_inference_service(db, s3)
            except Exception as e:
                raise build_model_not_initialized_error()

            embeddings = service.model.encode(documents.documents).tolist()

            return Embeddings(embeddings=embeddings)

    def _build_sentence_transformer_inference_service(
        self, db: Session, s3: Minio | None = None
    ):
        service = SentenceTransformerInferenceService(
            model_name=self.model_name,
            db=db,
            crud=self.crud_base,
            s3=s3,
            prioritize_internet_download=False,
        )

        service.initialize()
        return service
