from uuid import uuid4
from fastapi.encoders import jsonable_encoder
from typing import TypeVar
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..services.sentence_transformers_inference import (
    SentenceTransformersInferenceService,
)
from ..db.vector import DocumentModel, SourceModel
from .vector import CRUDVectorDocumentModel


VectorDocumentModelType = TypeVar("VectorDocumentModelType", bound=DocumentModel)
SourceModelType = TypeVar("SourceModelType", bound=SourceModel)

CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDSentenceTransformersDocumentModel(
    CRUDVectorDocumentModel[VectorDocumentModelType, CreateSchemaType, UpdateSchemaType]
):
    def create_and_calculate_single(
        self,
        db: Session,
        *,
        obj_in: CreateSchemaType,
        embedding_service: SentenceTransformersInferenceService
    ) -> VectorDocumentModelType:
        # just calls the multi with right args
        return self.create_and_calculate_multi(
            db=db, obj_in_list=[obj_in], embedding_service=embedding_service
        )[0]

    def create_and_calculate_multi(
        self,
        db: Session,
        *,
        obj_in_list: list[CreateSchemaType],
        embedding_service: SentenceTransformersInferenceService
    ) -> list[VectorDocumentModelType]:
        if not obj_in_list or len(obj_in_list) == 0:
            return []

        # ensure that page_content exists on schema definition
        if not hasattr(obj_in_list[0], "page_content"):
            raise ValueError(
                "obj_in_list objects must have a 'page_content' attribute."
            )

        obj_in_page_content_list = [obj_in.page_content for obj_in in obj_in_list]
        obj_in_embedding_list = embedding_service.model.encode(
            obj_in_page_content_list
        ).tolist()

        # add embedding, id, and convert to db object in a list
        db_obj_list = [
            self.model(
                **jsonable_encoder(obj_in),
                id=uuid4(),
                embedding=obj_in_embedding_list[i]
            )
            for i, obj_in in enumerate(obj_in_list)
        ]  # type: ignore

        db_obj_ids = [db_obj.id for db_obj in db_obj_list]
        db.add_all(db_obj_list)
        db.commit()
        return self.refresh_all_by_id(db, db_obj_ids=db_obj_ids)
