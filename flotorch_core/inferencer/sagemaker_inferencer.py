import random
import time
from typing import Any, Dict, List, Tuple
import boto3
from flotorch_core.inferencer.inferencer import BaseInferencer, DEFAULT_SYSTEM_PROMPT
from flotorch_core.logger.global_logger import get_logger
from flotorch_core.utils.sagemaker_utils import SageMakerUtils, INFERENCER_MODELS
from sagemaker.session import Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor

logger = get_logger()

class SageMakerInferencer(BaseInferencer):
    def __init__(self, model_id: str, region: str, role_arn: str, n_shot_prompts: int = 0, temperature: float = 0.7, n_shot_prompt_guide_obj: Dict[str, List[Dict[str, str]]] = None, max_tokens: int = 512, topP: int = 0.9):
        """
        Initialize the BedrockInferencer with Bedrock-specific parameters.

        Args:
            model_id (str): Identifier for the Bedrock model.
            region (str): AWS region where the Bedrock service is deployed.
            n_shot_prompts (int): Number of examples to include in few-shot learning.
            temperature (float): Sampling temperature for response generation.
            n_shot_prompt_guide_obj (Dict[str, List[Dict[str, str]]]): Guide object for few-shot examples.
        """
        super().__init__(model_id, region, n_shot_prompts, temperature, n_shot_prompt_guide_obj)
        self.role = role_arn
        self.client = boto3.client("sagemaker-runtime", region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.max_tokens = max_tokens
        self.topP = topP

        logger.info(f"Initializing SageMaker Generator for model: {model_id}")

        self.session = Session(boto_session=boto3.Session(region_name=region))

        self.inferencing_model_id = model_id
        self.inferencing_model_endpoint_name = f"{SageMakerUtils.sanitize_name(model_id)[:42]}-inferencing-endpoint"

        model_config = INFERENCER_MODELS.get(model_id)
        if not SageMakerUtils.check_endpoint_exists(self.sagemaker_client, self.inferencing_model_endpoint_name):
            if INFERENCER_MODELS[model_id]['model_source'] == 'jumpstart':
                SageMakerUtils.create_jumpstart_endpoint(self.sagemaker_client, model_config.get("instance_type"), self.region_name, self.role, model_id, self.inferencing_model_endpoint_name)
            elif INFERENCER_MODELS[model_id]['model_source'] == 'huggingface':
                SageMakerUtils.create_huggingface_endpoint(self.sagemaker_client, model_config.get("instance_type"), model_id, self.inferencing_model_endpoint_name, self.role, self.region_name)

        SageMakerUtils.wait_for_endpoint_creation(self.sagemaker_client, self.inferencing_model_endpoint_name)


        self.predictor = Predictor(
            endpoint_name=self.inferencing_model_endpoint_name,
            sagemaker_session=self.session
        )

        self.predictor.serializer = JSONSerializer()
        self.predictor.deserializer = JSONDeserializer()

        self.inferencing_predictor = self.predictor

    def generate_text(self, user_query: str, context: List[Dict], use_system: bool = True) -> Tuple[Dict[Any, Any], str]:
        if not self.inferencing_predictor:
            raise ValueError("Generation predictor not initialized")
        
        system_prompt, prompt = self.generate_prompt(user_query, use_system, context)

        payload = self.construct_payload(system_prompt, prompt)

        try:
            start_time = time.time()
            response = self.inferencing_predictor.predict(payload)
            latency = int((time.time() - start_time) * 1000)

            generated_text = self._extract_response(response)
            
            if "The final answer is:" in generated_text:
                answer = generated_text.split("The final answer is:")[1].strip()
            elif "Assistant:" in generated_text:
                answer = generated_text.split("Assistant:")[1].strip()
            else:
                answer = generated_text.strip()

            cleaned_response = self._clean_response(answer)

            if not cleaned_response or cleaned_response.isspace() or 'DRAFT' in cleaned_response:
                return None, "Unable to generate a proper response. Please try again."
            
            input_tokens = len(prompt) // 4
            output_tokens = len(generated_text) // 4
            total_tokens = input_tokens + output_tokens
            
            answer_metadata = {
                'inputTokens': input_tokens,
                'outputTokens': output_tokens,
                'totalTokens': total_tokens,
                'latencyMs': latency
            }
            
            return answer_metadata, cleaned_response
            

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _clean_response(self, text: str) -> str:
        """
        Cleans and formats the response text by removing common artifacts, ensuring proper sentence structure,
        and eliminating excessive whitespace or newlines.

        Args:
            text (str): The raw response text that needs to be cleaned.

        Returns:
            str: The cleaned and formatted response text.
        """
        artifacts = ['DRAFT', '[INST]', '[/INST]', 'Human:', 'Assistant:']
        
        cleaned_text = text.strip()
        for artifact in artifacts:
            cleaned_text = cleaned_text.replace(artifact, '').strip()

        sentence_endings = ['.', '!', '?']
        
        if not any(cleaned_text.rstrip().endswith(end) for end in sentence_endings):
            last_period = max(
                cleaned_text.rfind('.'),
                cleaned_text.rfind('!'),
                cleaned_text.rfind('?')
            )
            
            if last_period != -1:
                cleaned_text = cleaned_text[:last_period + 1]

        cleaned_text = ' '.join(cleaned_text.split())
        
        think_end_index = cleaned_text.find('</think>')
        if think_end_index != -1:
            cleaned_text = cleaned_text[think_end_index + len('</think>'):]

        return cleaned_text.strip()            

    def generate_prompt(self, user_query: str, use_system: bool, context: List[Dict]) -> Tuple[str, List[Dict[str, Any]]]:
        if self.n_shot_prompts < 0:
            raise ValueError("n_shot_prompt must be non-negative")
        
        # Get system prompt
        system_prompt = None
        if use_system:
            system_prompt = self.n_shot_prompt_guide_obj.get("system_prompt", "") if self.n_shot_prompt_guide_obj and self.n_shot_prompt_guide_obj.get("system_prompt") else DEFAULT_SYSTEM_PROMPT
        
        context_text = self.format_context(user_query, context)

        base_prompt = self.n_shot_prompt_guide_obj.get("user_prompt", "") if self.n_shot_prompt_guide_obj else ""

        if self.n_shot_prompts == 0:
            logger.info("into zero shot prompt")

            if self.inferencing_model_id == "huggingface-llm-falcon-7b-instruct-bf16":
                prompt = f"""Below are search results and a query. Create a concise summary.
                    Query: {user_query}
                    Search Results: {context_text}
                    Summary:"""
                return None, prompt
            
            prompt = f"Human: {system_prompt + chr(10) if system_prompt else ''}\n\n{context_text}\n\n{base_prompt}\n\nAssistant: The final answer is:"
            return None, prompt.strip()
        
        examples = self.n_shot_prompt_guide_obj['examples']
        selected_examples = (random.sample(examples, self.n_shot_prompts) 
                        if len(examples) > self.n_shot_prompts 
                        else examples)
        
        example_text = ""
        for example in selected_examples:
            example_text += f"- {example['example']}\n"

        logger.info(f"into {self.n_shot_prompts} shot prompt  with examples {len(selected_examples)}")

        if self.inferencing_model_id == "huggingface-llm-falcon-7b-instruct-bf16":
            prompt = f"""Below are search results and a query. Create a concise summary.
                Query: {user_query}
                Few examples:\n 
                {example_text}\n 
                Search Results: {context_text}
                Summary:"""
                
            return None, prompt
        
        prompt = f"Human: {system_prompt + chr(10) if system_prompt else ''}Few examples:\n{example_text}\n{context_text}\n\n{base_prompt}\n\nAssistant: The final answer is:"
        return None, prompt.strip()
    
    def format_context(self, user_query: str, context: List[Dict[str, str]]) -> str:
        """Format context documents into a single string."""
        formatted_context = f"Search Query: {user_query}\n"
        
        try:
            if context:
                formatted_context += "\nRelevant Passages:\n"
                for i, item in enumerate(context, 1):
                    content = None
                    if 'text' in item:
                        content = item['text']
                    elif '_source' in item and 'text' in item['_source']:
                        content = item['_source']['text']
                    
                    if not content:
                        continue

                    score = item.get('_score', 'N/A')
                    formatted_context += f"\nPassage {i} (Score: {score}):\n{content}\n"
            return formatted_context
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            formatted_context += "Error processing context"
            return formatted_context
        
    def construct_payload(self, system_prompt: str, prompt: str) -> dict:
        """
        Constructs the payload dictionary for model inference with the given prompts and default parameters.
        
        Args:
            system_prompt (str): The system-level prompt that guides the model's behavior
            prompt (str): The actual prompt/query to be sent to the model

        """
        # Define default parameters for controlling the model's text generation
        default_params = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.topP,
            "do_sample": True
            }
        # Construct the complete payload with prompt and generation parameters
        payload = {
            "inputs": prompt,
            "parameters": default_params
            }
        
        return payload
    
    def _extract_response(self, response: dict) -> str:
        """
        Parses the response from the model and extracts the generated text.

        Args:
            response (dict): The raw response from the model
        """
        # Handle different response formats (Falcon vs Llama)
        if isinstance(response, list):
            # Falcon-style response: Retrieve generated text from the list
            return response[0].get('generated_text', '') if response else ''
        elif isinstance(response, dict):
            return response.get('generated_text', '')
        else:
            raise ValueError(f"Unexpected response format: {type(response)}")