from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseInferencer(ABC):
    """
    Abstract base class for all inferencers.
    Defines the common interface and shared functionality for inferencers.
    """

    def __init__(self, model_id: str, region: str = "us-east-1", n_shot_prompts: int = 0, temperature: float = 0.7, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None):
        """
        Initialize the inferencer with required parameters.

        Args:
            model_id (str): Identifier for the model.
            region (str): AWS region or equivalent for the inferencing service.
            n_shot_prompts (int): Number of examples to include in few-shot learning. Defaults to 0.
            temperature (float): Sampling temperature for response generation. Defaults to 0.7.
            n_shot_prompt_guide_obj (Any): Guide object for few-shot examples. Defaults to None.
        """
        self.model_id = model_id
        self.region_name = region
        self.n_shot_prompts = n_shot_prompts
        self.temperature = temperature
        self.n_shot_prompt_guide_obj = n_shot_prompt_guide_obj

    @abstractmethod
    def generate_text(self, user_query: str, context: List[Dict]) -> Tuple[Dict[Any, Any], str]:
        """
        Generate a response based on the user query and context.

        Args:
            user_query (str): The question or input from the user.
            context (List[Dict]): Contextual data to assist in generating a response.
            default_prompt (str): Default prompt to guide the response generation.
            **kwargs: Additional parameters for customization.

        Returns:
            Tuple[Dict[Any, Any], str]: Metadata and the generated response text.
        """
        pass

    @abstractmethod
    def generate_prompt(self, user_query: str, context: List[Dict]) -> str:
        """
        Construct a prompt for the inferencer based on the user query and context.

        Args:
            default_prompt (str): Default prompt to guide the response generation.
            user_query (str): The question or input from the user.
            context (List[Dict]): Contextual data to assist in generating a response.

        Returns:
            str: The constructed prompt.
        """
        pass

    @abstractmethod
    def format_context(self, context: List[Dict[str, str]]) -> str:
        """
        Format context documents into a single string for inclusion in the prompt.

        Args:
            context (List[Dict[str, str]]): List of context documents.

        Returns:
            str: Formatted context as a single string.
        """
        pass

    def _extract_response(self, response: Dict) -> str:
        """
        Extract the generated text from the response.

        Args:
            response (Dict): Response from the inferencing service.

        Returns:
            str: Extracted text from the response.
        """
        raise NotImplementedError("Subclasses must implement _extract_response")
