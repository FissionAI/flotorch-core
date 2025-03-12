from typing import Dict, Optional, Any
import yaml


class GuardrailCreateConfig:
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
