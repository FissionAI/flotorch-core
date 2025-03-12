import json
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from sagemaker.session import Session

from flotorch_core.logger.global_logger import get_logger


logger = get_logger()

EMBEDDING_MODELS = {
    "huggingface-sentencesimilarity-bge-large-en-v1-5": {
        "model_name": "bge-large",
        "model_source": "jumpstart",
        "dimension": 1024,
        "instance_type": "ml.g5.2xlarge",
        "input_key": "text_inputs"
    },
    "huggingface-sentencesimilarity-bge-m3": {
        "model_name": "bge-m3",
        "model_source": "jumpstart",
        "dimension": 1024,
        "instance_type": "ml.g5.2xlarge",
        "input_key": "text_inputs"
    },
    "huggingface-textembedding-gte-qwen2-7b-instruct": {
        "model_name": "qwen",
        "model_source": "jumpstart",
        "dimension": 3584,
        "instance_type": "ml.g5.2xlarge",
        "input_key": "inputs"
    }
}


INFERENCER_MODELS = {
    "meta-textgeneration-llama-3-1-8b-instruct": {
        "model_source": "jumpstart",
        "instance_type": "ml.g5.2xlarge"
    },
    "huggingface-llm-falcon-7b-instruct-bf16": {
        "model_source": "jumpstart",
        "instance_type": "ml.g5.2xlarge"
    },
    "meta-textgeneration-llama-3-3-70b-instruct": {
        "model_source": "jumpstart",
        "instance_type": "ml.p4d.24xlarge"
    }
    ,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "model_source": "huggingface",
        "instance_type": "ml.g5.2xlarge"
    }
    ,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "model_source": "huggingface",
        "instance_type": "ml.g5.xlarge"
    }
    ,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "model_source": "huggingface",
        "instance_type": "ml.g5.xlarge"
    }
    ,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "model_source": "huggingface",
        "instance_type": "ml.g6e.12xlarge"
    }
}


class SageMakerUtils:
    @staticmethod
    def check_endpoint_exists(sagemaker_client, endpoint_name):        
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"Endpoint '{endpoint_name}' status: {status}")
            return status == 'InService' or status == 'Creating'  # Returns True if InService, False otherwise

        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            if error_code == 'ValidationException' and 'Could not find endpoint' in e.response['Error']['Message']:
                print(f"Endpoint '{endpoint_name}' does not exist.")
                return False  # Return False if the endpoint does not exist
            
            else:
                # Re-raise unexpected exceptions
                print(f"An unexpected error occurred: {e}")
                raise

    @staticmethod
    def create_jumpstart_endpoint(sagemaker_client, instance_type, region, role, model_id: str, endpoint_name: str) -> bool:
        """
        Creates a SageMaker endpoint for JumpStart models.
        Reuses existing model and endpoint configuration if available.

        Args:
            model_id (str): The model ID for the SageMaker JumpStart model.
            endpoint_name (str): The name of the SageMaker endpoint to be created.

        Returns:
            bool: True if the endpoint is successfully created, False otherwise.
        """
        try:
            boto_session = boto3.Session(region_name=region)
            sagemaker_session = sagemaker.Session(boto_session=boto_session)

            # Initialize JumpStart model
            model = JumpStartModel(
                role=role,
                model_id=model_id,
                sagemaker_session=sagemaker_session
            )

            # Check if the endpoint configuration exists
            try:
                sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=endpoint_name
                )
                logger.info(f"Endpoint configuration '{endpoint_name}' exists. Deploying endpoint.")

                # Deploy the endpoint using the existing configuration
                sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_name
                )
            except sagemaker_client.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ValidationException' and 'Could not find endpoint configuration' in e.response['Error']['Message']:
                    logger.info(f"Endpoint configuration '{endpoint_name}' does not exist. Deploying using model.deploy().")

                    # Use model.deploy to handle everything
                    model.deploy(
                        initial_instance_count=1,
                        instance_type=instance_type,
                        endpoint_name=endpoint_name,
                        accept_eula=True
                    )
                else:
                    logger.error(f"Error while checking endpoint configuration: {e}")
                    raise

            logger.info(f"Successfully created endpoint '{endpoint_name}' for model '{model_id}'.")
            return True

        except sagemaker_client.exceptions.ResourceLimitExceeded as e:
            logger.error(f"Resource limit exceeded while creating endpoint: {e}")
            return False

        except Exception as e:
            logger.error(f"Error while creating endpoint '{endpoint_name}' for model '{model_id}': {e}")
            return False


    @staticmethod
    def create_huggingface_endpoint(sagemaker_client, instance_type, model_id: str, endpoint_name: str, role: str, region_name: str) -> bool:
        """
        Creates a SageMaker endpoint for HuggingFace models.
        - If the endpoint config exists, it deploys using the existing configuration.
        - If not, it does a full model deployment.

        Args:
            sagemaker_client (boto3.client): Boto3 SageMaker client.
            instance_type (str): AWS instance type for deployment.
            model_id (str): Hugging Face model ID.
            endpoint_name (str): Name of the SageMaker endpoint.
            role (str): IAM role for SageMaker execution.
            region_name (str): AWS region.

        Returns:
            bool: True if the endpoint is successfully created, False otherwise.
        """
        try:
            session = Session(boto_session=boto3.Session(region_name=region_name))
            hub = {
                'HF_MODEL_ID': model_id,
                'SM_NUM_GPUS': json.dumps(1)
            }
            huggingface_model = HuggingFaceModel(
                image_uri=get_huggingface_llm_image_uri("huggingface", version="2.3.1", region=region_name),
                env=hub,
                role=role,
                sagemaker_session = session 
            )

            predictor = huggingface_model.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                container_startup_health_check_timeout=300,
            )
            
            return predictor
        
        except sagemaker_client.exceptions.ResourceLimitExceeded as e:
            logger.error(f"Resource limit exceeded while creating endpoint: {e}")
            return False

        except Exception as e:
            logger.error(f"Error while creating endpoint '{endpoint_name}' for model '{model_id}': {e}")
            return False

        
        
    @staticmethod
    def wait_for_endpoint_creation(sagemaker_client, endpoint_name: str, wait_interval: int = 5, timeout: int = 100000) -> bool:
        """
        Waits until the SageMaker endpoint is created successfully.

        Args:
            endpoint_name (str): The name of the SageMaker endpoint.
            wait_interval (int): Time (in seconds) to wait between status checks. Default is 30 seconds.
            timeout (int): Maximum time (in seconds) to wait for the endpoint creation. Default is 1800 seconds (30 minutes).

        Returns:
            bool: True if the endpoint is successfully created and in service, False if timed out or failed.
        """
        import time

        start_time = time.time()
        logger.info(f"Waiting for endpoint '{endpoint_name}' to be in service...")

        try:
            while time.time() - start_time < timeout:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response["EndpointStatus"]

                logger.info(f"Endpoint '{endpoint_name}' status: {status}")

                if status == "InService":
                    logger.info(f"Endpoint '{endpoint_name}' is now in service.")
                    return True
                elif status == "Failed":
                    logger.error(f"Endpoint '{endpoint_name}' creation failed.")
                    return False

                logger.error(f"waiting for endpoint creating'{endpoint_name}' , status: {status}")
                time.sleep(wait_interval)  # Wait before checking again

            logger.error(f"Timeout while waiting for endpoint '{endpoint_name}' to be created.")
            return False

        except sagemaker_client.exceptions.ResourceNotFound:
            logger.error(f"Endpoint '{endpoint_name}' not found. Ensure the creation process has started.")
            return False

        except Exception as e:
            logger.error(f"Error while checking endpoint status for '{endpoint_name}': {e}")
            return False
    
    @staticmethod
    def sanitize_name( name: str) -> str:
        """Sanitize the endpoint name to follow AWS naming conventions"""
        # Replace any character that's not alphanumeric or hyphen with hyphen
        import re
        name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
        # Ensure it starts with a letter
        if not name[0].isalpha(): 
            name = 'n' + name
        # Truncate to 63 characters (AWS limit)
        return name[:63]