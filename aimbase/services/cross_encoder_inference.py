from .sentence_transformer_inference import SentenceTransformerInferenceService


class CrossEncoderInferenceService(SentenceTransformerInferenceService):
    def import_dynamic_dependencies(self):
        """
        Overrides default to import CrossEncoder
        """

        from sentence_transformers import CrossEncoder

        self.sentence_transformer_class = CrossEncoder

    def cache_load_by_type(self):
        """
        Overrides default to instantiate and return CrossEncoder model from cache.
        Errors handled by parent class.
        """

        # explicitly pass local cache path to load
        return self.sentence_transformer_class(model_name=self.get_model_cache_path())

    def download_by_type(self):
        """
        Overrides default to download, instantiate, save, and return CrossEncoder model.
        Errors handled by parent class.
        """

        # download model to cache and return instantiated model
        model = self.sentence_transformer_class(model_name=self.model_name)
        model.save(path=self.get_model_cache_path())
        return model
