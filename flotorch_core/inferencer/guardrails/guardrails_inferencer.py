from typing import Any, Dict, List, Tuple
from flotorch_core.guardrails.guardrails import BaseGuardRail
from flotorch_core.inferencer.inferencer import BaseInferencer


class GuardRailsInferencer(BaseInferencer):
    def __init__(self, base_inferencer: BaseInferencer, base_guardrail: BaseGuardRail):
        self.base_inferencer = base_inferencer
        self.base_guardrail = base_guardrail

    def generate_text(self, user_query: str, context: List[Dict]) -> Tuple[Dict[Any, Any], str]:
        metadata, answer = self.base_inferencer.generate_text(user_query, context)

        guardrail_response = self.base_guardrail.apply_guardrail(answer, 'OUTPUT')
        if guardrail_response['action'] == 'GUARDRAIL_INTERVENED':
            return {
                'guardrail_output_assessment': guardrail_response.get('assessments', []),
                'guardrail_blocked': True
            }, guardrail_response['outputs'][0]['text']
        
        return metadata, answer
    
    def generate_prompt(self, user_query: str, context: List[Dict]) -> str:
        return self.base_inferencer.generate_prompt(user_query, context)

    def format_context(self, context: List[Dict[str, str]]) -> str:
        return self.base_inferencer.format_context(context)