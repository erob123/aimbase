import os
import traceback
from typing import BinaryIO
from .config import get_aimbase_settings
from instarest import LogConfig
from fastapi import HTTPException
from minio.error import InvalidResponseError
from minio import Minio
import hashlib
from pathlib import Path
from tqdm import tqdm

logger = LogConfig(LOGGER_NAME="minio.py").build_logger()


# TODO: test hash calculation same both directions and for local folder calculation
def build_client():
    if not get_aimbase_settings().minio_region:
        return Minio(
            get_aimbase_settings().minio_endpoint_url,
            access_key=get_aimbase_settings().minio_access_key,
            secret_key=get_aimbase_settings().minio_secret_key,
            secure=get_aimbase_settings().minio_secure,
        )

    return Minio(
        get_aimbase_settings().minio_endpoint_url,
        access_key=get_aimbase_settings().minio_access_key,
        secret_key=get_aimbase_settings().minio_secret_key,
        secure=get_aimbase_settings().minio_secure,
        region=get_aimbase_settings().minio_region,
    )


def download_folder_from_minio(s3: Minio, folder_path: str) -> str:
    # Create a directory for the downloaded files
    local_folder_path_obj = Path(folder_path)
    local_folder_path_obj.mkdir(parents=True, exist_ok=True)

    # Pull the folder name from the path
    folder_name = get_folder_name_from_path(folder_path)

    # Calculate the SHA256 hash while streaming and downloading files
    hash_object = hashlib.sha256()

    try:
        # List all objects in the bucket with the given prefix (folder_name)
        objects_generator = s3.list_objects(
            bucket_name=get_aimbase_settings().minio_bucket_name,
            prefix=folder_name,
            recursive=True,
        )

        # Convert the generator output by list_objects() to a list
        objects = list(objects_generator)

        # Sort the objects to ensure a deterministic order of processing files
        sorted_objects = sorted(objects, key=lambda obj: obj.object_name)

        for obj in sorted_objects:
            filename = obj.object_name

            # Use parent folder path because minio object name includes the folder name
            local_file_path_obj = local_folder_path_obj.parent / filename
            logger.debug(f"Downloading file {filename} to {local_file_path_obj}")

            # Download and write the file contents to disk while updating the hash
            data = s3.get_object(
                bucket_name=get_aimbase_settings().minio_bucket_name,
                object_name=filename,
            )

            # Create the parent directories if they don't exist
            local_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(local_file_path_obj, "wb") as f:
                for chunk in tqdm(data.stream(32 * 1024), unit="B", unit_scale=True):
                    f.write(chunk)
                    hash_object.update(chunk)

    except InvalidResponseError as e:
        logger.error(f"Failed to download file from Minio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    finally:
        # Ensure data.close() and data.release_conn() always execute, even if an error occurs
        data.close()
        data.release_conn()

    # Get the final SHA256 hash value
    hex_dig = hash_object.hexdigest()

    return hex_dig


# TODO: change to multipart upload and calculate hash while uploading
def upload_folder_to_minio(s3: Minio, folder_path: str) -> str:
    # Create a Path object for the folder
    folder_path_obj = Path(folder_path)

    # Pull the folder name from the path
    folder_name = get_folder_name_from_path(folder_path)

    # Calculate the SHA256 hash while streaming and uploading files
    hash_object = hashlib.sha256()

    try:
        # Iterate through each file in the folder
        for file_path in folder_path_obj.rglob("*"):
            if file_path.is_file():
                object_name = "".join(
                    [folder_name, "/", str(file_path.relative_to(folder_path_obj))]
                )

                with open(file_path, "rb") as f:
                    file_size = file_path.stat().st_size

                    try:
                        s3.put_object(
                            bucket_name=get_aimbase_settings().minio_bucket_name,
                            object_name=object_name,
                            data=f,
                            length=file_size,
                            content_type="application/octet-stream",
                        )

                    except InvalidResponseError as e:
                        logger.error(f"Failed to upload file to Minio: {e}")
                        raise HTTPException(
                            status_code=500, detail="Internal Server Error"
                        ) from e

                    # Update the hash
                    # TODO: use tqdm to show progress bar with multipart upload vs. just hashing
                    f.seek(0)  # Reset the cursor to the beginning of the file
                    with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
                        while chunk := f.read(32 * 1024):
                            hash_object.update(chunk)
                            pbar.update(len(chunk))

    except Exception as e:
        logger.error(f"An error occurred while uploading folder to Minio: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    # Get the final SHA256 hash value
    hex_dig = hash_object.hexdigest()

    return hex_dig


def get_folder_name_from_path(path):
    """
    Extracts the final folder name from a given path.

    Args:
        path (str): The input path.

    Returns:
        str: The folder name.
    """
    # Normalize the path to handle different path separators
    normalized_path = os.path.normpath(path)

    # Split the path into its components
    path_components = normalized_path.split(os.sep)

    # Return the last component, which is the last subpath
    return path_components[-1]


def calculate_folder_hash(folder_path: str) -> str:
    hash_object = hashlib.sha256()

    try:
        files = sorted(item for item in Path(folder_path).rglob("*") if item.is_file())
        for item in files:
            with open(item, "rb") as f:
                while chunk := f.read(32 * 1024):
                    hash_object.update(chunk)

    except Exception as e:
        print(f"An error occurred while calculating hash: {e}")
        raise

    return hash_object.hexdigest()
