import pytest
from unittest.mock import Mock, patch

from flotorch_core.inferencer.inferencer_provider_factory import InferencerProviderFactory
from flotorch_core.inferencer.bedrock_inferencer import BedrockInferencer
from flotorch_core.inferencer.gateway_inferencer import GatewayInferencer
from flotorch_core.inferencer.sagemaker_inferencer import SageMakerInferencer

# === Fixtures ===

@pytest.fixture
def factory():
    return InferencerProviderFactory()

@pytest.fixture
def mock_bedrock_client():
    with patch('boto3.client') as mock_client:
        mock_bedrock = Mock()
        mock_client.return_value = mock_bedrock
        yield mock_bedrock

@pytest.fixture
def mock_sagemaker_client():
    with patch('boto3.client') as mock_client:
        mock_sagemaker = Mock()
        # Describe endpoint returns a real dict instead of a mock
        mock_sagemaker.describe_endpoint.return_value = {
            'EndpointStatus': 'InService'
        }
        mock_client.return_value = mock_sagemaker
        yield mock_sagemaker


# === Test cases ===

def test_create_bedrock_inferencer(factory, mock_bedrock_client):
    """Should return a BedrockInferencer instance"""
    with patch('boto3.client', return_value=mock_bedrock_client):
        inferencer = factory.create_inferencer_provider(
            gateway_enabled=False,
            base_url='http://test-url',
            api_key='test-key',
            service='bedrock',
            model_id='anthropic.claude-v2',
            region='us-east-1',
            arn_role='test-role'
        )
        assert isinstance(inferencer, BedrockInferencer)
        assert inferencer.model_id == 'anthropic.claude-v2'

def test_create_sagemaker_inferencer(factory, mock_sagemaker_client):
    """Should return a SageMakerInferencer instance"""
    with patch('boto3.client', return_value=mock_sagemaker_client):
        inferencer = factory.create_inferencer_provider(
            gateway_enabled=False,
            base_url='http://test-url',
            api_key='test-key',
            service='sagemaker',
            model_id='sagemaker-model',
            region='us-west-2',
            arn_role='arn:aws:iam::123456789012:role/test-role'
        )
        assert isinstance(inferencer, SageMakerInferencer)
        assert inferencer.model_id == 'sagemaker-model'

def test_create_gateway_inferencer(factory):
    """Should return a GatewayInferencer instance when gateway is enabled"""
    inferencer = factory.create_inferencer_provider(
        gateway_enabled=True,
        base_url='http://test-gateway',
        api_key='gateway-key',
        service='bedrock',  # service is ignored when gateway is enabled
        model_id='gateway-model',
        region='us-west-1',
        arn_role='test-role'
    )
    assert isinstance(inferencer, GatewayInferencer)
    assert inferencer.model_id == 'gateway-model'

def test_invalid_service_raises_value_error(factory):
    """Should raise ValueError when an invalid service is passed"""
    with pytest.raises(ValueError, match="Unsupported service scheme: invalid_service"):
        factory.create_inferencer_provider(
            gateway_enabled=False,
            base_url='http://test-url',
            api_key='test-key',
            service='invalid_service',
            model_id='some-model',
            region='us-east-1',
            arn_role='test-role'
        )

@pytest.mark.parametrize("gateway_enabled,service,expected_class", [
    (True, 'bedrock', GatewayInferencer),
    (True, 'sagemaker', GatewayInferencer),
    (False, 'bedrock', BedrockInferencer),
    (False, 'sagemaker', SageMakerInferencer),
])
def test_factory_class_combinations(factory, mock_bedrock_client, mock_sagemaker_client,
                                    gateway_enabled, service, expected_class):
    """Should return correct inferencer class based on gateway and service combination"""
    with patch('boto3.client') as mock_boto3:
        if service == 'bedrock':
            mock_boto3.return_value = mock_bedrock_client
        else:
            mock_boto3.return_value = mock_sagemaker_client

        inferencer = factory.create_inferencer_provider(
            gateway_enabled=gateway_enabled,
            base_url='http://some-url',
            api_key='some-key',
            service=service,
            model_id='model-x',
            region='us-east-1',
            arn_role='arn:aws:iam::123456789012:role/test-role'
        )
        assert isinstance(inferencer, expected_class)

def test_missing_required_params_raises_type_error(factory):
    """Should raise TypeError when required params are missing"""
    # Missing base_url
    with pytest.raises(TypeError):
        factory.create_inferencer_provider(
            gateway_enabled=True,
            api_key='key',
            service='gateway',
            model_id='model',
            region='us-east-1',
            arn_role='role'
        )

    # Missing api_key
    with pytest.raises(TypeError):
        factory.create_inferencer_provider(
            gateway_enabled=True,
            base_url='http://url',
            service='gateway',
            model_id='model',
            region='us-east-1',
            arn_role='role'
        )

    # Missing arn_role for non-gateway sagemaker
    with pytest.raises(TypeError):
        factory.create_inferencer_provider(
            gateway_enabled=False,
            base_url='http://url',
            api_key='key',
            service='sagemaker',
            model_id='model',
            region='us-east-1'
        )

def test_create_inferencer_with_optional_params(factory, mock_bedrock_client):
    """Should create inferencer with optional parameters"""
    with patch('boto3.client', return_value=mock_bedrock_client):
        inferencer = factory.create_inferencer_provider(
            gateway_enabled=False,
            base_url='http://test-url',
            api_key='test-key',
            service='bedrock',
            model_id='anthropic.claude-v2',
            region='us-east-1',
            arn_role='test-role',
            n_shot_prompts=2,
            temperature=0.8,
            n_shot_prompt_guide_obj={'test': [{'input': 'test', 'output': 'test'}]}
        )
        assert inferencer.n_shot_prompts == 2
        assert inferencer.temperature == 0.8
        assert inferencer.n_shot_prompt_guide_obj == {'test': [{'input': 'test', 'output': 'test'}]}


