from .base_evaluator import BaseEvaluator
from typing import Dict, Type

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class EvaluatorServiceError(Exception):
    """Custom exception for inference service related errors"""
    pass

class EvalFactory:

    _registry: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register_evaluator(cls, service_type: str, llm_service: str, evaluator_cls: Type[BaseEvaluator]):
        key = f"{service_type}:{llm_service}"
        cls._registry[key] = evaluator_cls

    @classmethod
    def create_evaluator(cls, region: str, eval_embedding_model: str, eval_retrieval_model: str, knowledge_base: bool, eval_service: str, gateway_enabled: bool, gateway_url: str, gateway_api_key: str, temperature: float = 0.4):
        
        
        llm_service = 'custom_gateway' if gateway_enabled else 'bedrock'
        
        key = f"{eval_service}:{llm_service}"
        
        evaluator_cls = cls._registry.get(key)
        if not evaluator_cls:
            raise EvaluatorServiceError(f"No evaluator_cls registered for service {eval_service} and type {llm_service}")
        
        if gateway_enabled:
            return evaluator_cls(
                region,
                temperature,
                eval_embedding_model,
                eval_retrieval_model,
                knowledge_base,
                gateway_url,
                gateway_api_key
                )
        else:
            return evaluator_cls(
                region,
                temperature,
                eval_embedding_model, 
                eval_retrieval_model,
                knowledge_base
                )


eval_factory = EvalFactory()

def register(service_type, llm_service):
    def decorator(cls):
        eval_factory.register_evaluator(service_type, llm_service, cls)
        return cls
    return decorator