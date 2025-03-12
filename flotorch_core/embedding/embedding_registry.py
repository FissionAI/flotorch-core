class EmbeddingRegistry:
    def __init__(self):
        self._models = {}

    def register_model(self, model_id, embedding_class):
        self._models[model_id] = embedding_class

    def get_model(self, model_id):
        embedding_class = self._models.get(model_id)
        if not embedding_class:
            raise ValueError(f"Model '{model_id}' not found in the registry.")
        return embedding_class

# Global registry instance
embedding_registry = EmbeddingRegistry()

def register(model_id):
    def decorator(cls):
        embedding_registry.register_model(model_id, cls)
        return cls
    return decorator