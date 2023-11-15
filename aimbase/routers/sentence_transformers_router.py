from datetime import datetime
from fastapi import Depends, HTTPException
from pydantic import BaseModel, StrictFloat
from ..crud.sentence_transformers_document import CRUDSentenceTransformersDocumentModel
from ..db.base import BaseAIModel, FineTunedAIModel
from ..services.cross_encoder_inference import CrossEncoderInferenceService
from ..services.sentence_transformers_inference import (
    SentenceTransformersInferenceService,
)
from ..dependencies import get_minio
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
    cross_encoder_name: str = "cross-encoder/ms-marco-TinyBERT-L-6"
    crud_base: CRUDSentenceTransformersDocumentModel
    crud_ai_base: BaseAIModel | FineTunedAIModel

    # override and do not call super() to prevent default CRUD endpoints
    def _add_endpoints(self):
        self._define_encode()
        self._define_create_documents()
        self._define_knn_search()

    def _define_encode(self):
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
                raise self._build_model_not_initialized_error()

            embeddings = service.model.encode(documents.documents).tolist()

            return Embeddings(embeddings=embeddings)

    def _define_create_documents(self):
        # CREATE MULTIPLE DOCUMENTS WITH EMBEDDING CALCULATION
        @self.router.post(
            "/create_and_embed_multi",
            response_model=list[self.schema_base.Entity],
            responses=self.responses,
            summary=f"Create multiple new documents with calculation",
            response_description=f"List of created documents with calculated embeddings",
        )
        async def create_and_embed_multi(
            documents: list[self.schema_base.EntityCreate],
            db: Session = Depends(get_db),
            s3: Minio | None = Depends(get_minio),
        ) -> list[self.schema_base.Entity]:
            try:
                service = self._build_sentence_transformer_inference_service(db, s3)
            except Exception as e:
                raise self._build_model_not_initialized_error()

            # Create and calculate embeddings for multiple documents
            created_documents: list[
                self.schema_base.get_model_type()
            ] = self.crud_base.create_and_calculate_multi(
                db, obj_in_list=documents, embedding_service=service
            )

            return created_documents

    def _define_knn_search(self):
        class RankedNeighbor(BaseModel):
            document: self.schema_base.Entity
            score: StrictFloat

        # kNN SEARCH
        @self.router.post(
            "/knn_search",
            response_model=list[RankedNeighbor],
            responses=self.responses,
            summary=f"kNN search for similar documents",
            response_description=f"List of documents similar to the query",
        )
        async def knn_search(
            query: str,
            k: int = 100,  # number of nearest neighbors to return
            title: str | None = None,
            downloaded_datetime_start: datetime | None = None,
            downloaded_datetime_end: datetime | None = None,
            similarity_measure: str = "cosine_distance",
            db: Session = Depends(get_db),
            s3: Minio | None = Depends(get_minio),
        ) -> list[self.schema_base.Entity]:
            try:
                embedding_service = self._build_sentence_transformer_inference_service(
                    db, s3
                )
                cross_encoder_service = self._build_cross_encoder_inference_service(
                    db, s3
                )
            except Exception as e:
                raise self._build_model_not_initialized_error()

            # Calculate embedding for the query
            query_embedding = embedding_service.model.encode(query)

            # Perform kNN search
            retrieved_documents = (
                self.crud_base.get_by_source_metadata_and_nearest_neighbors(
                    db,
                    title=title,
                    downloaded_datetime_start=downloaded_datetime_start,
                    downloaded_datetime_end=downloaded_datetime_end,
                    vector_query=query_embedding,
                    k=k,
                    similarity_measure=similarity_measure,
                )
            )

            # Step 2: Score the documents via cross encoder
            cross_encoder_inputs = [
                [query, doc.page_content] for doc in retrieved_documents
            ]
            scores = cross_encoder_service.model.predict(cross_encoder_inputs)

            # Step 3: Sort the scores in decreasing order
            unsorted_neighbors = []
            for doc, score in zip(retrieved_documents, scores):
                unsorted_neighbors.append(RankedNeighbor(document=doc, score=score))

            reranked_neighbors = sorted(
                unsorted_neighbors, key=lambda item: item.score, reverse=True
            )
            return reranked_neighbors

    def _build_model_not_initialized_error(self):
        return HTTPException(
            status_code=500,
            detail=f"{self.model_name} is not initialized",
        )

    def _build_sentence_transformer_inference_service(
        self, db: Session, s3: Minio | None = None
    ):
        service = SentenceTransformersInferenceService(
            model_name=self.model_name,
            db=db,
            crud=self.crud_ai_base,
            s3=s3,
            prioritize_internet_download=False,
        )

        service.initialize()
        return service

    def _build_cross_encoder_inference_service(
        self, db: Session, s3: Minio | None = None
    ):
        service = CrossEncoderInferenceService(
            model_name=self.cross_encoder_name,
            db=db,
            crud=self.crud_ai_base,
            s3=s3,
            prioritize_internet_download=False,
        )

        service.initialize()
        return service
