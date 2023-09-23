import shutil
import traceback
from typing import Any
from aimbase.core.minio import calculate_folder_hash
from aimbase.services.base import BaseAIInferenceService


# TODO: pull tracebacks out of all except blocks and handle at app level
class SentenceTransformerInferenceService(BaseAIInferenceService):
    # internal only
    sentence_transformer_class: Any | None = None

    def dev_init(self):
        """
        Initialize the service object for development purposes.
        """

        # init imports and download model from internet
        self.initialize()

        # upload model to minio
        self.upload_model_to_minio()

        # delete the model from the cache and all files within directory
        shutil.rmtree(self.get_model_cache_path())

    def initialize(self):
        try:
            self.import_dynamic_dependencies()
        except ImportError:
            msg = (
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`"
            )

            self.logger.error(msg)
            traceback.print_exc()
            raise ImportError(msg)

        super().initialize()

    def load_model_from_cache(self):
        """
        Load the model from the cache into self.model.
        """

        try:
            self.model = self.cache_load_by_type()
        except:
            msg = "Model not found in cache."
            self.logger.error(msg)
            traceback.print_exc()

            raise Exception(msg)

    def download_model_internet(self):
        """
        Download the model from the internet and cache it locally.
        Returns the SHA256 hash of the downloaded model.
        """

        self.download_by_type()

        # calculate sha256 hash of model folder
        return calculate_folder_hash(self.get_model_cache_path())

    def import_dynamic_dependencies(self):
        """
        Import the specific class needed from sentence-transformers package for this model.
        This default is for SentenceTransformer.
        Overriden by child classes as needed.
        """

        from sentence_transformers import SentenceTransformer

        self.sentence_transformer_class = SentenceTransformer

    def cache_load_by_type(self):
        """
        Specific cache load for the sentence_transformer_class being initialized.
        This default is for SentenceTransformer.
        Must return model loaded from local cache given self.model.
        Does not need to handle errors, that is done elsewhere.

        Returns the intantiated model.

        Overriden by child classes as needed.
        """

        # the default SentenceTransformer class loads from cache first
        return self.download_by_type()

    def download_by_type(self):
        """
        Specific download for the sentence_transformer_class being initialized.
        This default is for SentenceTransformer.
        Must download from the internet and save the model in torch
        format in self.get_model_cache_path().

        Returns the intantiated model.

        Overriden by child classes as needed.
        """

        # download model to cache and return instantiated model
        return self.sentence_transformer_class(
            model_name_or_path=self.model_name, cache_folder=self.get_model_cache_path()
        )
