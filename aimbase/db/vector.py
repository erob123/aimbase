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
    source_id = Column(UUID, ForeignKey("sourcemodel.id"))
    source = relationship("SourceModel")
    type = Column(String(), nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "DocumentModel",
        "polymorphic_on": "type",
    }


class AllMiniDocumentModel(DocumentModel):
    embedding = Column(Vector(384))

    __mapper_args__ = {
        "polymorphic_identity": "AllMiniDocumentModel",
    }


class SourceModel(DeclarativeBase):
    model_name = Column(String(), unique=True)
    local_cache_path = Column(String())
    sha256 = Column(String(64))
    uploaded_minio = Column(Boolean(), default=False)


class FineTunedAIModel(DeclarativeBase):
    # model name does not need to be unique because we can have multiple fine tuned models of the same base model
    model_name = Column(String())
    local_cache_path = Column(String())
    sha256 = Column(String(64))
    uploaded_minio = Column(Boolean(), default=False)
    version_sequence = Sequence(
        __qualname__.lower() + "_version_sequence"
    )  # see here for autoincrementing versioning: https://copyprogramming.com/howto/using-sqlalchemy-orm-for-a-non-primary-key-unique-auto-incrementing-id
    version = Column(
        Integer,
        version_sequence,
        server_default=version_sequence.next_value(),
        index=True,
        unique=True,
        nullable=False,
    )
    type = Column(String(100), nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "FineTunedAIModel",
        "polymorphic_on": "type",
    }


class FineTunedAIModelWithBaseModel(FineTunedAIModel):
    id = Column(UUID, ForeignKey("finetunedaimodel.id"), primary_key=True)
    base_ai_model_id = Column(UUID, ForeignKey("baseaimodel.id"))
    base_ai_model = relationship("BaseAIModel")

    __mapper_args__ = {
        "polymorphic_identity": "FineTunedAIModelWithBaseModel",
    }
