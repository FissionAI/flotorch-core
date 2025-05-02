class EmbeddingRegistry:
    """
    A registry for embedding models.
    This registry allows for the registration and retrieval of embedding models
    using a unique identifier (model_id).
    """
    def __init__(self):
        """
        Initializes the EmbeddingRegistry with an empty dictionary to hold models.
        """
        self._models = {}

    def register_model(self, model_id, embedding_class):
        """
        Registers an embedding model with a unique identifier.
        Args:
            model_id (str): A unique identifier for the embedding model.
            embedding_class (type): The class of the embedding model to register.
        Returns:
            None
        """
        self._models[model_id] = embedding_class

    def get_model(self, model_id):
        """
        Retrieves an embedding model class by its unique identifier.
        Args:
            model_id (str): The unique identifier for the embedding model.
        Returns:
            type: The class of the embedding model.
        """
        embedding_class = self._models.get(model_id)
        if not embedding_class:
            raise ValueError(f"Model '{model_id}' not found in the registry.")
        return embedding_class

# Global registry instance
embedding_registry = EmbeddingRegistry()

def register(model_id):
    """
    A decorator to register an embedding model with a unique identifier.
    Args:
        model_id (str): A unique identifier for the embedding model.
    Returns:
        function: The decorator function.
    """
    def decorator(cls):
        embedding_registry.register_model(model_id, cls)
        return cls
    return decorator