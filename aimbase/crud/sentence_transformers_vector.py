from uuid import uuid4
from typing import TypeVar
from pydantic import BaseModel
from sqlalchemy.orm import Session
from instarest import DeclarativeBase
from ..services.sentence_transformers_inference import (
    SentenceTransformersInferenceService,
)
from .vector import CRUDVectorStore, CRUDDocument
from ..db.vector import DocumentModel


VectorStoreType = TypeVar("VectorStoreType", bound=DeclarativeBase)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDSentenceTransformersVectorStore(
    CRUDVectorStore[VectorStoreType, CreateSchemaType, UpdateSchemaType]
):
    def __init__(self, model: type[VectorStoreType]):
        super().__init__(model=model)
        self.crud_document = CRUDDocument(DocumentModel)

    def create_and_calculate_single(
        self,
        db: Session,
        *,
        obj_in: BaseModel,  # must use DocumentSchema defined by SchemaBase or equivalent pydantic representation
        embedding_service: SentenceTransformersInferenceService
    ) -> VectorStoreType:
        # just calls the multi with right args
        return self.create_and_calculate_multi(
            db=db, docs_in_list=[obj_in], embedding_service=embedding_service
        )[0]

    def create_and_calculate_multi(
        self,
        db: Session,
        *,
        docs_in_list: list[
            BaseModel
        ],  # must use DocumentSchema defined by SchemaBase or equivalent pydantic representation
        embedding_service: SentenceTransformersInferenceService
    ) -> list[VectorStoreType]:
        if not docs_in_list or len(docs_in_list) == 0:
            return []

        # ensure that page_content exists on schema definition
        if not hasattr(docs_in_list[0], "page_content"):
            raise ValueError(
                "obj_in_list objects must have a 'page_content' attribute."
            )

        # save underlying documents first
        new_documents_db = self.crud_document.create_all_using_id(
            db=db, obj_in_list=docs_in_list
        )

        # calculate embeddings
        new_docs_page_content_list = [
            new_doc.page_content for new_doc in new_documents_db
        ]
        new_docs_embedding_list = embedding_service.model.encode(
            new_docs_page_content_list
        ).tolist()

        # add embedding, id, and convert to db object in a list
        db_obj_list = [
            self.model(
                id=uuid4(), embedding=new_docs_embedding_list[i], document_id=new_doc.id
            )
            for i, new_doc in enumerate(new_documents_db)
        ]  # type: ignore

        db_obj_ids = [db_obj.id for db_obj in db_obj_list]
        db.add_all(db_obj_list)
        db.commit()
        return self.refresh_all_by_id(db, db_obj_ids=db_obj_ids)
