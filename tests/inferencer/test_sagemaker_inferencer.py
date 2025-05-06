import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError
from flotorch_core.inferencer.sagemaker_inferencer import SageMakerInferencer

MOCK_ROLE_ARN = "arn:aws:iam::123456789012:role/test-role"

@pytest.fixture
def sagemaker_inferencer():
    with patch('flotorch_core.inferencer.sagemaker_inferencer.boto3.client') as mock_boto3_client, \
         patch('flotorch_core.utils.sagemaker_utils.SageMakerUtils.check_endpoint_exists', return_value=True):

        mock_sagemaker_client = Mock()
        mock_runtime_client = Mock()

        mock_sagemaker_client.describe_endpoint.return_value = {
            'EndpointStatus': 'InService'
        }

        mock_runtime_client.invoke_endpoint.return_value = {
            'Body': MagicMock(
                read=lambda: json.dumps({"generated_text": "test response"}).encode()
            )
        }

        def boto3_client_side_effect(service_name, *args, **kwargs):
            if service_name == "sagemaker":
                return mock_sagemaker_client
            elif service_name == "sagemaker-runtime":
                return mock_runtime_client
            else:
                raise ValueError(f"Unknown service_name: {service_name}")

        mock_boto3_client.side_effect = boto3_client_side_effect

        return SageMakerInferencer(
            model_id="test-model",
            region="us-east-1",
            role_arn=MOCK_ROLE_ARN
        )

class TestSageMakerInferencer:
    def test_initialization(self):
        with patch("flotorch_core.inferencer.sagemaker_inferencer.boto3.client") as mock_boto3:
            mock_sagemaker = Mock()
            mock_sagemaker.describe_endpoint.return_value = {
                'EndpointStatus': 'InService'
            }
            mock_boto3.return_value = mock_sagemaker

            with patch('flotorch_core.utils.sagemaker_utils.SageMakerUtils.check_endpoint_exists', return_value=True):
                inferencer = SageMakerInferencer(
                    model_id="test-model",
                    region="us-east-1",
                    role_arn=MOCK_ROLE_ARN
                )
                assert inferencer.model_id == "test-model"
                assert inferencer.region_name == "us-east-1"
                assert inferencer.role == MOCK_ROLE_ARN
                assert inferencer.inferencing_model_endpoint_name == "test-model-inferencing-endpoint"

    def test_initialization_with_custom_values(self):
        with patch("flotorch_core.inferencer.sagemaker_inferencer.boto3.client") as mock_boto3:
            mock_sagemaker = Mock()
            mock_sagemaker.describe_endpoint.return_value = {
                'EndpointStatus': 'InService'
            }
            mock_boto3.return_value = mock_sagemaker

            with patch("flotorch_core.utils.sagemaker_utils.SageMakerUtils.check_endpoint_exists", return_value=True):
                inferencer = SageMakerInferencer(
                    model_id="test-model",
                    region="us-west-2",
                    role_arn=MOCK_ROLE_ARN,
                    n_shot_prompts=2,
                    temperature=0.5
                )
                assert inferencer.n_shot_prompts == 2
                assert inferencer.temperature == 0.5

   

    def test_generate_prompt(self, sagemaker_inferencer):
        context = [{"text": "test context"}]
        prompt = sagemaker_inferencer.generate_prompt(
            user_query="test query",
            context=context
        )
        assert isinstance(prompt, str)
        assert "test query" in prompt
        assert "test context" in prompt

    def test_format_context(self, sagemaker_inferencer):
        query = "test query"
        context = [
            {"text": "context 1"},
            {"text": "context 2"}
        ]
        formatted = sagemaker_inferencer.format_context(query, context)
        assert isinstance(formatted, str)
        assert "test query" in formatted
        assert "context 1" in formatted
        assert "context 2" in formatted

        
    def test_generate_text(self):
        with patch("flotorch_core.inferencer.sagemaker_inferencer.boto3.client") as mock_boto3_client, \
            patch("flotorch_core.utils.sagemaker_utils.SageMakerUtils.check_endpoint_exists", return_value=True), \
            patch("flotorch_core.inferencer.sagemaker_inferencer.Predictor") as mock_predictor_class:
            
            # Mock clients
            mock_sagemaker = Mock()
            mock_runtime = Mock()

            # Set up mock predictor
            mock_predictor_instance = Mock()
            mock_predictor_instance.predict.return_value = {"generated_text": "test response"}
            mock_predictor_class.return_value = mock_predictor_instance

            # Simulating a valid endpoint description
            mock_sagemaker.describe_endpoint.return_value = {
                'EndpointStatus': 'InService'
            }

            # Mock the boto3 client side effect
            def boto3_side_effect(service_name, *args, **kwargs):
                if service_name == "sagemaker":
                    return mock_sagemaker
                elif service_name == "sagemaker-runtime":
                    return mock_runtime
                raise ValueError(f"Unknown service {service_name}")

            mock_boto3_client.side_effect = boto3_side_effect

            # Initialize the SageMakerInferencer
            inferencer = SageMakerInferencer(
                model_id="test-model",
                region="us-east-1",
                role_arn=MOCK_ROLE_ARN
            )

            # Call the method under test
            result = inferencer.generate_text(
                user_query="test query",
                context=[{"text": "test context"}]
            )
            
            # Verify that we get a tuple with metadata and text
            assert isinstance(result, tuple)
            metadata, response = result
            
            # Validate the response
            assert isinstance(metadata, dict)
            assert isinstance(response, str)
            assert "test response" in response

    def test_extract_response(self, sagemaker_inferencer):
        # Since _extract_response is not implemented in SageMakerInferencer, 
        # we should test _clean_response instead
        response = "DRAFT test response Human: Assistant:"
        cleaned = sagemaker_inferencer._clean_response(response)
        assert cleaned == "test response"

    def test_invoke_endpoint_error(self, sagemaker_inferencer):
        with patch.object(sagemaker_inferencer, 'inferencing_predictor') as mock_predictor:
            mock_predictor.predict.side_effect = Exception("Test exception")
            
            # Should return error string, not raise exception
            result = sagemaker_inferencer.generate_text(
                user_query="test query", 
                context=[{"text": "test context"}]
            )
            
            assert isinstance(result, str)
            assert "Error generating response" in result

    def test_invalid_response_format(self, sagemaker_inferencer):
        with patch.object(sagemaker_inferencer, 'inferencing_predictor') as mock_predictor:
            mock_predictor.predict.return_value = "invalid format"
            
            # Should return error string, not raise exception
            result = sagemaker_inferencer.generate_text(
                user_query="test query", 
                context=[{"text": "test context"}]
            )
            
            assert isinstance(result, str)
            assert "Error generating response" in result

    def test_with_n_shot_prompts(self):
        with patch("flotorch_core.inferencer.sagemaker_inferencer.boto3.client") as mock_boto3:
            mock_sagemaker = Mock()
            mock_sagemaker.describe_endpoint.return_value = {
                'EndpointStatus': 'InService'
            }
            mock_boto3.return_value = mock_sagemaker

            with patch("flotorch_core.utils.sagemaker_utils.SageMakerUtils.check_endpoint_exists", return_value=True):
                n_shot_guide = {
                    "examples": [
                        {"example": "test input 1 -> test output 1"},
                        {"example": "test input 2 -> test output 2"}
                    ]
                }
                inferencer = SageMakerInferencer(
                    model_id="test-model",
                    region="us-east-1",
                    role_arn=MOCK_ROLE_ARN,
                    n_shot_prompts=2,
                    n_shot_prompt_guide_obj=n_shot_guide
                )
                prompt = inferencer.generate_prompt(
                    user_query="test query",
                    context=[{"text": "test context"}]
                )
                assert "test input 1" in prompt
                assert "test output 1" in prompt
                assert "test input 2" in prompt
                assert "test output 2" in prompt

        

        @pytest.mark.parametrize("endpoint_status", [
            'Creating', 'Updating', 'SystemUpdating', 'RollingBack', 'Failed',
        ])
        def test_endpoint_status_checks(self, endpoint_status):
            with patch("flotorch_core.inferencer.sagemaker_inferencer.boto3.client") as mock_boto3:
                mock_sagemaker = Mock()
                mock_sagemaker.describe_endpoint.return_value = {
                    'EndpointStatus': endpoint_status
                }
                mock_boto3.return_value = mock_sagemaker

                with patch("flotorch_core.utils.sagemaker_utils.SageMakerUtils.check_endpoint_exists", return_value=False):
                    with pytest.raises(Exception):
                        SageMakerInferencer(
                            model_id="test-model",
                            region="us-east-1",
                            role_arn=MOCK_ROLE_ARN
                        )
