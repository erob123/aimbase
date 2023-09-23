from .core.minio import build_client


# ********use for minio initialization*****
def get_minio():
    return build_client()
