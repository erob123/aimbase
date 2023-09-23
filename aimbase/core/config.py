from typing import TypeVar

from pydantic import Field, validator
from instarest.core.config import (
    CoreSettings,
    EnvironmentSettings,
    set_core_settings,
)

aimbase_environment_settings = None  # :meta private:
aimbase_settings = None  # :meta private:


class AimbaseSettings(CoreSettings):
    """
    Settings specific to aimbase.
    """

    minio_bucket_name: str = ""
    minio_endpoint_url: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_region: str = ""
    minio_secure: bool = True

    openai_api_key: str = ""

    # validator to remove http:// or https:// from the minio_undpoint_url
    @validator("minio_endpoint_url", pre=True, always=True)
    def remove_http_or_https(cls, v: str) -> str:
        if v.startswith("http://"):
            return v[len("http://") :]
        if v.startswith("https://"):
            return v[len("https://") :]
        return v


AimbaseSettingsType = TypeVar("AimbaseSettingsType", bound=AimbaseSettings)


# make it possible to load AimbaseSettings from environment variables and to inherit from
# AimbaseEnvironmentSettings with generic settings type
class AimbaseEnvironmentSettings(EnvironmentSettings[AimbaseSettingsType]):
    def pull_settings(self, settings_type=AimbaseSettings) -> AimbaseSettingsType:
        return super().pull_settings(settings_type)


def set_aimbase_settings(
    new_aimbase_environment_settings: AimbaseEnvironmentSettings,
) -> None:
    """
    Set the environment settings and aimbase settings objects.
    """
    global aimbase_environment_settings, aimbase_settings

    aimbase_environment_settings = new_aimbase_environment_settings
    aimbase_settings = aimbase_environment_settings.pull_settings()

    # make sure to set the instarest settings as well
    set_core_settings(aimbase_environment_settings)


def get_aimbase_settings() -> AimbaseSettings:
    """
    Get the aimbase settings object.
    """
    global aimbase_settings

    if aimbase_settings is None:
        raise ValueError(
            "Aimbase Settings not initialized.  Please call set_aimbase_settings() first."
        )

    return aimbase_settings


def get_aimbase_environment_settings() -> AimbaseEnvironmentSettings:
    """
    Get the aimbase environment settings object.
    """
    global aimbase_environment_settings

    if aimbase_environment_settings is None:
        raise ValueError(
            "Aimbase Settings not initialized.  Please call set_aimbase_settings() first."
        )

    return aimbase_environment_settings


aimbase_environment_settings = AimbaseEnvironmentSettings()
set_aimbase_settings(aimbase_environment_settings)
