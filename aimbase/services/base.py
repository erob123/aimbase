import os
from typing import Any
from pydantic import BaseModel, validator
from aimbase.core.constants import MODEL_CACHE_BASEDIR
from aimbase.crud.base import CRUDBaseAIModel
from aimbase.db.base import BaseAIModel
from minio import Minio
from sqlalchemy.orm import Session
from pathlib import Path
from aimbase.core.minio import download_folder_from_minio, upload_folder_to_minio
from instarest import LogConfig
from logging import Logger


# TODO: add docker-compose with minio services
# TODO: add SHA256 hash to model cache path and minio folder
class BaseAIInferenceService(BaseModel):
    # one of sha256 or model_name must be provided
    sha256: str | None = None
    model_name: str | None = None

    # db and crud objects must be provided
    db: Session
    crud: CRUDBaseAIModel

    # optional objects, must be provided if minio is used
    s3: Minio | None = None
    prioritize_internet_download: bool = True

    # internal objects, not to be used by external callers
    db_object: BaseAIModel | None = None
    model: Any | None = None
    initialized: bool = False
    logger: Logger | None = None

    class Config:
        arbitrary_types_allowed = True

    # pydantic validator to ensure that one of sha256 or model_name is provided
    @validator("model_name", pre=True, always=True)
    def either_sha256_or_model_name(cls, v, values):
        if v is None and values["sha256"] is None:
            raise ValueError("Either 'sha256' or 'model_name' must be provided.")
        return v

    @validator("logger", pre=True, always=True)
    def set_logger(cls, v):
        return v or LogConfig(LOGGER_NAME=cls.__name__).build_logger()

    def dev_init(self):
        """
        Initialize the service object for development purposes.
        """

        raise NotImplementedError("dev_init() not implemented for this service.")

    def initialize(self):
        """
        Initialize the service object by downloading the model from Minio and/or the internet, and caching it locally.
        """

        if self.sha256 is not None:
            self.db_object = self.get_obj_by_sha256()

            # throw error if db_object is None and sha256 is provided
            if self.db_object is None:
                raise ValueError("SHA256 not found in the database.")

            # validate that model name on db_object matches self.model_name if self.model_name is not None
            if (
                self.model_name is not None
                and self.db_object.model_name != self.model_name
            ):
                raise ValueError(
                    f"Model name {self.model_name} does not match model name {self.db_object.model_name} on DB object"
                )

            if self.model_name is None:
                self.model_name = self.db_object.model_name

        elif self.model_name is not None:
            self.db_object = self.get_obj_by_model_name()

        # TODO: verify
        model_cache_path = self.get_model_cache_path()

        # try to load the model from the cache & close out if successful
        # only if already initialized in the db, else must download
        if self.db_object and os.path.isdir(model_cache_path):
            try:
                self.load_model_from_cache()
                self.initialized = True
                return
            except:
                pass

        # Create the directory if it doesn't exist
        Path(model_cache_path).mkdir(parents=True, exist_ok=True)

        model_hash = self.download_model()

        # if self.sha256 is not None, validate that the hash matches
        if self.sha256 is None:
            self.sha256 = model_hash
        elif self.sha256 != model_hash:
            raise ValueError(
                f"SHA256 hash {self.sha256} does not match downloaded model hash {model_hash}"
            )

        # initialize new db object if it doesn't exist
        # NOTE: only works if self.model_name is provided, sha256
        # option assumes DB entry already exists since that is the
        # only way to get the sha256
        if self.db_object is None:
            unsaved_db_object = BaseAIModel(
                model_name=self.model_name,
                local_cache_path=self.get_model_cache_path(),
                sha256=self.sha256,
                uploaded_minio=False,
            )
            self.db_object = self.crud.create(self.db, obj_in=unsaved_db_object)

        self.load_model_from_cache()
        self.initialized = True

    def get_model_cache_path(self):
        """
        Get the path to the model cache.
        """
        if self.model_name is None:
            raise ValueError("Model name not set.")
        else:
            return os.path.join(MODEL_CACHE_BASEDIR, self.model_name)

    def download_model(self):
        """
        Download the model from the internet or Minio and cache it locally.
        Prioritizes Minio if prioritize_internet_download is False, but
        will try both ways.

        Returns the SHA256 hash of the downloaded model.
        """

        try:
            if self.prioritize_internet_download:
                # try internet download first
                try:
                    return self.download_model_internet()
                except:
                    # if that fails, try Minio
                    return self.download_model_from_minio()
            else:
                # try Minio first
                try:
                    return self.download_model_from_minio()
                except:
                    # if that fails, try internet
                    return self.download_model_internet()
        except:
            raise ValueError("Could not download model from internet or Minio.")

    def download_model_from_minio(self):
        """
        Download the model from Minio and cache it locally.
        Returns the SHA256 hash of the downloaded model.
        Do not override this method.
        """

        # throw error if s3 is None
        if self.s3 is None:
            raise ValueError("Minio client is not set.")

        model_cache_path = self.get_model_cache_path()

        # download the model folder from minio
        self.logger.info(f"Downloading model from Minio to {model_cache_path}")
        model_hash = download_folder_from_minio(
            s3=self.s3, folder_path=model_cache_path
        )
        self.logger.info(f"Downloaded model from Minio to {model_cache_path}")

        return model_hash

    def upload_model_to_minio(self):
        """
        Upload the model to Minio.
        Returns the SHA256 hash of the uploaded model.
        Do not override this method.
        """

        # throw error if s3 is None
        if self.s3 is None:
            raise ValueError("Minio client is not set.")

        model_cache_path = self.get_model_cache_path()

        # upload the model folder to minio
        self.logger.info(f"Uploading model from {model_cache_path} to Minio")
        model_hash = upload_folder_to_minio(s3=self.s3, folder_path=model_cache_path)
        self.logger.info(f"Uploaded model from {model_cache_path} to Minio")

        return model_hash

    def load_model_from_cache(self):
        """
        Load the model from the cache into self.model.
        Overriden by child classes as needed.
        """

        self.model = "replace me"
        raise NotImplementedError(
            "load_model_from_cache() not implemented for this service."
        )

    def get_obj_by_sha256(self):
        """
        Get the DB object by SHA256.
        Overriden by child classes as needed.
        """

        return self.crud.get_by_sha256(self.db, sha256=self.sha256)

    def get_obj_by_model_name(self):
        """
        Get the DB object by model_name.
        Overriden by child classes as needed.
        """

        return self.crud.get_by_model_name(self.db, model_name=self.model_name)

    def download_model_internet(self):
        """
        Download the model from the internet and cache it locally.
        Returns the SHA256 hash of the downloaded model.
        Overriden by child classes as needed.
        """

        raise NotImplementedError(
            "download_model_internet() not implemented for this service."
        )


# TODO: add upsert / training service that keeps minio and DB in line
## if model name and using FT model, pull latest version if minio
# save base models in minio as default_bucket/model_name/<any files>
#
