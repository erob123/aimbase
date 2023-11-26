from enum import Enum as CoreEnum
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    UUID,
    String,
    Boolean,
    Sequence,
    Enum,
    DateTime,
    ARRAY,
)
from sqlalchemy.orm import relationship
from instarest import DeclarativeBase
from pgvector.sqlalchemy import Vector


class TagsEnum(str, CoreEnum):
    TAG1 = "Tag 1"
    TAG2 = "Tag 2"
    TAG3 = "Tag 3"


class SourceModel(DeclarativeBase):
    tags = Column(ARRAY(Enum(TagsEnum), dimensions=1, as_tuple=True))
    title = Column(String())
    description = Column(String())
    downloaded_datetime = Column(DateTime)
    private_url = Column(String())
    public_url = Column(String())
    embedding = Column(Vector(384))


class DocumentModel(DeclarativeBase):
    page_content = Column(String())
    source_id = Column(UUID, ForeignKey("sourcemodel.id"), nullable=True)
    source = relationship("SourceModel")


##########Vector Stores by Model (Separate tables since different models have different dimensions)##########
###Ensure that you define an "embedding" column if using with CRUDVectorStore####


# Document store for https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
class AllMiniVectorStore(DeclarativeBase):
    embedding = Column(Vector(384))
    document_id = Column(UUID, ForeignKey("documentmodel.id"))
    document = relationship("DocumentModel")
