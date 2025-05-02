from typing import Dict, Optional, Any
import yaml


class GuardrailCreateConfig:
    """
    GuardrailCreateConfig is a class that represents the configuration for creating a guardrail.
    It includes various policies and filters that can be applied to the guardrail.
    """
    def __init__(
            self,
            name: str,
            description: str,
            content_policy: Optional[Dict[str, Any]] = None,
            topic_policy: Optional[Dict[str, Any]] = None,
            word_policy: Optional[Dict[str, Any]] = None,
            sensitive_info_policy: Optional[Dict[str, Any]] = None,
            contextual_grounding_policy: Optional[Dict[str, Any]] = None,
            input_filter: Optional[Dict[str, Any]] = None,
            output_filter: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the GuardrailCreateConfig with the given parameters.
        Args:
            name (str): The name of the guardrail.
            description (str): A description of the guardrail.
            content_policy (Optional[Dict[str, Any]]): Content policy for the guardrail.
            topic_policy (Optional[Dict[str, Any]]): Topic policy for the guardrail.
            word_policy (Optional[Dict[str, Any]]): Word policy for the guardrail.
            sensitive_info_policy (Optional[Dict[str, Any]]): Sensitive information policy for the guardrail.
            contextual_grounding_policy (Optional[Dict[str, Any]]): Contextual grounding policy for the guardrail.
            input_filter (Optional[Dict[str, Any]]): Input filter for the guardrail.
            output_filter (Optional[Dict[str, Any]]): Output filter for the guardrail.
        Returns:
            None
        """
        self.name = name
        self.description = description
        self.content_policy = content_policy
        self.topic_policy = topic_policy
        self.word_policy = word_policy
        self.sensitive_info_policy = sensitive_info_policy
        self.contextual_grounding_policy = contextual_grounding_policy
        self.input_filter = input_filter
        self.output_filter = output_filter

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the GuardrailCreateConfig instance to a dictionary.
        Returns:
            Dict[str, Any]: A dictionary representation of the GuardrailCreateConfig instance.
        """
        return {
            "name": self.name,
            "description": self.description,
            "content_policy": self.content_policy,
            "topic_policy": self.topic_policy,
            "word_policy": self.word_policy,
            "sensitive_info_policy": self.sensitive_info_policy,
            "contextual_grounding_policy": self.contextual_grounding_policy,
            "input_filter": self.input_filter,
            "output_filter": self.output_filter
        }

    @staticmethod
    def from_yaml(yaml_file: str) -> 'GuardrailCreateConfig':
        """
        Creates a GuardrailCreateConfig instance from a YAML file.
        Args:
            yaml_file (str): The path to the YAML file.
        Returns:
            GuardrailCreateConfig: An instance of GuardrailCreateConfig.
        """
        with open(yaml_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return GuardrailCreateConfig(
            name=config_data.get('name'),
            description=config_data.get('description'),
            content_policy=config_data.get('content_policy'),
            topic_policy=config_data.get('topic_policy'),
            word_policy=config_data.get('word_policy'),
            sensitive_info_policy=config_data.get('sensitive_info_policy'),
            contextual_grounding_policy=config_data.get('contextual_grounding_policy'),
            input_filter=config_data.get('input_filter'),
            output_filter=config_data.get('output_filter')
        )
