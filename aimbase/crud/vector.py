from datetime import datetime
from typing import TypeVar
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, Query
from instarest import CRUDBase
from ..db.vector import DocumentModel, SourceModel


VectorDocumentModelType = TypeVar("VectorDocumentModelType", bound=DocumentModel)
SourceModelType = TypeVar("SourceModelType", bound=SourceModel)

CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDVectorDocumentModel(
    CRUDBase[VectorDocumentModelType, CreateSchemaType, UpdateSchemaType]
):
    def __init__(self, model: type[VectorDocumentModelType]):
        """
        Check that the model has an "embedding" column before initializing.
        """

        if "embedding" not in model.__table__.columns:
            raise ValueError(
                "CRUDVectorDocumentModel must be initialized with a model that has an 'embedding' column."
            )

        super().__init__(model=model)

    def get_by_source_metadata(
        self,
        db: Session,
        *,
        title: str | None = None,
        downloaded_datetime_start: datetime | None = None,
        downloaded_datetime_end: datetime | None = None,
    ) -> list[VectorDocumentModelType]:
        return self.get_by_source_metadata_and_nearest_neighbors(
            db=db,
            title=title,
            downloaded_datetime_start=downloaded_datetime_start,
            downloaded_datetime_end=downloaded_datetime_end,
        )

    def get_by_nearest_neighbors(
        self,
        db: Session,
        *,
        vector_query: list[float] | None = None,
        k: int = 100,  # number of nearest neighbors to return
        similarity_measure: str = "cosine_distance",
    ) -> list[VectorDocumentModelType]:
        return self.get_by_source_metadata_and_nearest_neighbors(
            db=db,
            vector_query=vector_query,
            k=k,
            similarity_measure=similarity_measure,
        )

    def get_by_source_metadata_and_nearest_neighbors(
        self,
        db: Session,
        *,
        title: str | None = None,
        downloaded_datetime_start: datetime | None = None,
        downloaded_datetime_end: datetime | None = None,
        vector_query: list[float] | None = None,
        k: int = 100,  # number of nearest neighbors to return
        similarity_measure: str = "cosine_distance",
    ) -> list[VectorDocumentModelType]:
        db_query = db.query(self.model)
        db_query = self._filter_by_source_metadata(
            db_query=db_query,
            title=title,
            downloaded_datetime_start=downloaded_datetime_start,
            downloaded_datetime_end=downloaded_datetime_end,
        )
        db_query = self._filter_by_nearest_neighbors(
            db_query=db_query,
            vector_query=vector_query,
            k=k,
            similarity_measure=similarity_measure,
        )

        return db_query.all()

    ############################ PRIVATE METHODS ############################
    def _filter_by_nearest_neighbors(
        self,
        db_query: Query,
        vector_query: list[float] | None = None,
        k: int = 100,  # number of nearest neighbors to return
        similarity_measure: str = "cosine_distance",
    ) -> Query:
        if vector_query:
            if similarity_measure == "cosine_distance":
                db_query = db_query.order_by(
                    self.model.embedding.cosine_distance(vector_query)
                ).limit(k)
            elif similarity_measure == "l2_distance":
                db_query = db_query.order_by(
                    self.model.embedding.l2_distance(vector_query)
                ).limit(k)
            elif similarity_measure == "max_inner_product":
                db_query = db_query.order_by(
                    self.model.embedding.max_inner_product(vector_query)
                ).limit(
                    k
                )  # still same syntax since uses negative inner product
            else:
                raise ValueError(
                    "Invalid similarity measure. Supported measures: 'cosine_distance', 'l2_distance', 'max_inner_product'."
                )

        return db_query

    def _filter_by_source_metadata(
        self,
        db_query: Query,
        titles: list[str] | None = None,
        downloaded_datetime_start: datetime | None = None,
        downloaded_datetime_end: datetime | None = None,
    ) -> Query:
        if titles:
            title_filters = [
                func.lower(self.model.source.title) == title.lower() for title in titles
            ]
            db_query = db_query.filter(or_(*title_filters))

        if downloaded_datetime_start:
            db_query = db_query.filter(
                self.model.source.downloaded_datetime >= downloaded_datetime_start
            )

        if downloaded_datetime_end:
            db_query = db_query.filter(
                self.model.source.downloaded_datetime <= downloaded_datetime_end
            )

        return db_query
