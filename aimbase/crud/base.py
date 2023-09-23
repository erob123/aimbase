from typing import TypeVar
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session, Query
from instarest import DeclarativeBase, CRUDBase
from ..db.base import BaseAIModel, FineTunedAIModel, FineTunedAIModelWithBaseModel

BaseAIModelType = TypeVar("BaseAIModelType", bound=BaseAIModel)
FineTunedAIModelType = TypeVar("FineTunedAIModelType", bound=FineTunedAIModel)
FineTunedAIModelWithBaseModelType = TypeVar(
    "FineTunedAIModelWithBaseModelType", bound=FineTunedAIModelWithBaseModel
)

CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBaseAIModel(CRUDBase[BaseAIModelType, CreateSchemaType, UpdateSchemaType]):
    def get_by_sha256(self, db: Session, *, sha256: str) -> BaseAIModelType | None:
        return _get_by_sha256(self, db, sha256=sha256)

    def get_by_model_name(
        self, db: Session, *, model_name: str
    ) -> BaseAIModelType | None:
        if not isinstance(model_name, str):
            return None

        # model_name is unique so we can use .first()
        return db.query(self.model).filter(self.model.model_name == model_name).first()

    def get_minio_uploaded_by_model_name(
        self, db: Session, *, model_name: str
    ) -> BaseAIModelType | None:
        if not isinstance(model_name, str):
            return None

        # model_name is unique so we can use .first()
        return (
            db.query(self.model)
            .filter(
                self.model.model_name == model_name, self.model.uploaded_minio == True
            )
            .first()
        )


class CRUDFineTunedAIModel(
    CRUDBase[FineTunedAIModelType, CreateSchemaType, UpdateSchemaType]
):
    def get_by_sha256(self, db: Session, *, sha256: str) -> FineTunedAIModelType | None:
        return _get_by_sha256(self, db, sha256=sha256)

    def get_latest_version_by_model_name(
        self, db: Session, *, model_name: str
    ) -> FineTunedAIModelType | None:
        return _get_latest_version_by_model_name(self, db, model_name=model_name)

    def get_latest_minio_uploaded_version_by_model_name(
        self, db: Session, *, model_name: str
    ) -> FineTunedAIModelType | None:
        return _get_latest_minio_uploaded_version_by_model_name(
            self, db, model_name=model_name
        )


class CRUDFineTunedAIModelWithBaseModel(
    CRUDBase[FineTunedAIModelWithBaseModelType, CreateSchemaType, UpdateSchemaType]
):
    def get_by_sha256(
        self, db: Session, *, sha256: str
    ) -> FineTunedAIModelWithBaseModelType | None:
        return _get_by_sha256(self, db, sha256=sha256)

    def get_latest_version_by_model_name(
        self, db: Session, *, model_name: str
    ) -> FineTunedAIModelWithBaseModelType | None:
        return _get_latest_version_by_model_name(self, db, model_name=model_name)

    def get_latest_minio_uploaded_version_by_model_name(
        self, db: Session, *, model_name: str
    ) -> FineTunedAIModelWithBaseModelType | None:
        return _get_latest_minio_uploaded_version_by_model_name(
            self, db, model_name=model_name
        )


def _get_by_sha256(
    crud_instance, db: Session, *, sha256: str
) -> DeclarativeBase | None:
    if not sha256:
        return None

    return (
        db.query(crud_instance.model)
        .filter(crud_instance.model.sha256 == sha256)
        .first()
    )


def _get_latest_version_by_model_name(
    crud_instance, db: Session, *, model_name: str
) -> DeclarativeBase | None:
    # see here for info: https://stackoverflow.com/questions/30784456/sqlalchemy-return-a-record-filtered-by-max-value-of-a-column
    subqry = db.query(func.max(crud_instance.model.version)).filter(
        crud_instance.model.model_name == model_name,
    )

    return _get_version_filtered_by_model_name(
        crud_instance, db, model_name=model_name, subqry=subqry
    )


def _get_latest_minio_uploaded_version_by_model_name(
    crud_instance, db: Session, *, model_name: str
) -> DeclarativeBase | None:
    # see here for info: https://stackoverflow.com/questions/30784456/sqlalchemy-return-a-record-filtered-by-max-value-of-a-column
    subqry = db.query(func.max(crud_instance.model.version)).filter(
        crud_instance.model.model_name == model_name,
        crud_instance.model.uploaded == True,
    )

    return _get_version_filtered_by_model_name(
        crud_instance, db, model_name=model_name, subqry=subqry
    )


def _get_version_filtered_by_model_name(
    crud_instance, db: Session, *, model_name: str, subqry: Query
) -> DeclarativeBase | None:
    if not isinstance(model_name, str):
        return None

    return (
        db.query(crud_instance.model)
        .filter(
            crud_instance.model.model_name == model_name,
            crud_instance.model.version == subqry,
        )
        .first()
    )
