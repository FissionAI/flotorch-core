from abc import ABC, abstractmethod
import boto3

class BaseGuardRail(ABC):
    """
    Base class for guardrails. This class is used to define the interface for guardrails.
    """    

    def __init__(self, prompt=True, response=True):
        """
        Initialize the guardrail.
        Args:
            prompt (bool): Whether to apply guardrail to the prompt.
            response (bool): Whether to apply guardrail to the response.
        """
        self.prompt = prompt
        self.response = response
        
    @abstractmethod
    def apply_guardrail(self, text: str,
        source: str = 'INPUT'):
        """
        Apply the guardrail to the text.
        Args:
            text (str): The text to apply the guardrail to.
            source (str): The source of the text.
        """
        pass

class BedrockGuardrail(BaseGuardRail):
    """
    Class for applying guardrails using AWS Bedrock.
    """
    def __init__(self, guardrail_id: str, guardrail_version: str, region_name: str = 'us-east-1', runtime_client = None):
        """
        Initialize the BedrockGuardrail.
        Args:
            guardrail_id (str): The ID of the guardrail.
            guardrail_version (str): The version of the guardrail.
            region_name (str): The AWS region name.
            runtime_client: The Bedrock runtime client.
        """
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.runtime_client = runtime_client or boto3.client('bedrock-runtime', region_name=region_name)
        
    def apply_guardrail(self, text: str,
        source: str = 'INPUT'):
        """
        Apply the guardrail to the text.
        Args:
            text (str): The text to apply the guardrail to.
            source (str): The source of the text.
        Returns:
            response: The response from the Bedrock runtime client.
        """
        try:
            request_params = {
                'guardrailIdentifier': self.guardrail_id,
                'guardrailVersion': self.guardrail_version,
                'source': source,
                'content': [{"text": {"text": text}}]
            }
            response = self.runtime_client.apply_guardrail(**request_params)
            return response
        except Exception as e:
            print(f"Error applying guardrail: {str(e)}")
            raise