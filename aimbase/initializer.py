from instarest import LogConfig
from minio import Minio
from minio.error import InvalidResponseError
from aimbase.core.config import get_aimbase_environment_settings, get_aimbase_settings
from aimbase.core.minio import build_client


# ST fine tuned ai service, initialization, single crud router with encoding
# chainbase (specific tasks with models...here or another package?  start with query retrieval and marco rerank)
class AimbaseInitializer:
    def __init__(self):
        self.aimbase_logger = LogConfig(
            LOGGER_NAME=self.__class__.__name__
        ).build_logger()

    def init_minio_bucket(self, s3: Minio) -> None:
        bucket_name = get_aimbase_settings().minio_bucket_name
        try:
            if not s3.bucket_exists(bucket_name):
                s3.make_bucket(bucket_name)
        except InvalidResponseError as err:
            self.aimbase_logger.error(err)

    def execute(self, migration_toggle=False) -> None:
        # environment can be one of 'local', 'development, 'test', 'staging', 'production'
        environment = get_aimbase_environment_settings().environment

        # setup minio client if available (i.e., not in unit tests)
        if environment in ["local", "development", "staging", "production"]:
            self.aimbase_logger.info("Connecting MinIO client")
            s3 = build_client()
            self.aimbase_logger.info("MinIO client connected")

        if environment in ["local", "development"]:
            self.aimbase_logger.info("Setting up MinIO bucket")
            self.init_minio_bucket(s3)
            self.aimbase_logger.info("MinIO bucket set up.")
