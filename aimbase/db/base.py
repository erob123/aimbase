from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    UUID,
    String,
    Boolean,
    Sequence,
)
from sqlalchemy.orm import relationship
from instarest import DeclarativeBase


# TODO: Are local_cache_path and uploaded_minio needed?  If not, remove them.
class BaseAIModel(DeclarativeBase):
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
