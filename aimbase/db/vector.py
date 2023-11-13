from sqlalchemy import (
    Column,
    ForeignKey,
    UUID,
    String,
    DateTime,
)
from sqlalchemy.orm import relationship
from instarest import DeclarativeBase
from pgvector.sqlalchemy import Vector


class SourceModel(DeclarativeBase):
    title = Column(String())
    description = Column(String())
    downloaded_datetime = Column(DateTime)
    private_url = Column(String())
    public_url = Column(String())
    embedding = Column(Vector(384))


class DocumentModel(DeclarativeBase):
    page_content = Column(String())
    source_id = Column(UUID, ForeignKey("sourcemodel.id"))
    source = relationship("SourceModel")
    type = Column(String(), nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "DocumentModel",
        "polymorphic_on": "type",
    }


##########Document Stores by Model (Separate tables since different models have different dimensions)##########
###Ensure that you define an "embedding" column if using with CRUDVectorDocumentModel  ####


# Document store for https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
class AllMiniDocumentModel(DocumentModel):
    embedding = Column(Vector(384))

    __mapper_args__ = {
        "polymorphic_identity": "AllMiniDocumentModel",
    }
