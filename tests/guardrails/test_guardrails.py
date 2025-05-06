import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError
from flotorch_core.guardrails.guardrails import BaseGuardRail, BedrockGuardrail

@pytest.fixture(autouse=True)
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    with patch.dict('os.environ', {
        'AWS_ACCESS_KEY_ID': 'testing',
        'AWS_SECRET_ACCESS_KEY': 'testing',
        'AWS_SECURITY_TOKEN': 'testing',
        'AWS_SESSION_TOKEN': 'testing',
        'AWS_DEFAULT_REGION': 'us-east-1'
    }):
        yield

@pytest.fixture
def mock_boto3_client():
    """Fixture for mocked boto3 client"""
    with patch('boto3.client') as mock_client:
        mock_runtime = Mock()
        mock_client.return_value = mock_runtime
        yield mock_runtime

@pytest.fixture
def guardrail_config():
    """Fixture for basic guardrail configuration"""
    return {
        'guardrail_id': 'test-guardrail-id',
        'guardrail_version': '1.0',
        'region_name': 'us-east-1'
    }

@pytest.fixture
def bedrock_guardrail(mock_boto3_client, guardrail_config):
    """Fixture for BedrockGuardrail instance"""
    return BedrockGuardrail(
        guardrail_id=guardrail_config['guardrail_id'],
        guardrail_version=guardrail_config['guardrail_version'],
        region_name=guardrail_config['region_name'],
        runtime_client=mock_boto3_client
    )

def test_base_guardrail_is_abstract():
    """Test that BaseGuardRail cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseGuardRail()

def test_bedrock_guardrail_initialization(guardrail_config):
    """Test BedrockGuardrail initialization with different configurations"""
    with patch('boto3.client') as mock_client:
        guardrail = BedrockGuardrail(
            guardrail_id=guardrail_config['guardrail_id'],
            guardrail_version=guardrail_config['guardrail_version']
        )
        
        assert guardrail.guardrail_id == guardrail_config['guardrail_id']
        assert guardrail.guardrail_version == guardrail_config['guardrail_version']

def test_bedrock_guardrail_custom_region(guardrail_config):
    """Test BedrockGuardrail with custom region"""
    with patch('boto3.client') as mock_client:
        custom_region = 'us-west-2'
        guardrail = BedrockGuardrail(
            guardrail_id=guardrail_config['guardrail_id'],
            guardrail_version=guardrail_config['guardrail_version'],
            region_name=custom_region
        )
        
        assert guardrail.guardrail_id == guardrail_config['guardrail_id']
        assert guardrail.guardrail_version == guardrail_config['guardrail_version']

def test_bedrock_guardrail_success(bedrock_guardrail, mock_boto3_client):
    """Test successful guardrail application"""
    expected_response = {
        'action': 'PASS',
        'results': [{'status': 'OK'}]
    }
    mock_boto3_client.apply_guardrail.return_value = expected_response

    response = bedrock_guardrail.apply_guardrail("test content")
    
    assert response['action'] == 'PASS'
    mock_boto3_client.apply_guardrail.assert_called_once()

def test_bedrock_guardrail_failure(bedrock_guardrail, mock_boto3_client):
    """Test guardrail failure handling"""
    error_response = {
        'Error': {
            'Code': 'InternalServerException',
            'Message': 'AWS Error'
        }
    }
    mock_boto3_client.apply_guardrail.side_effect = ClientError(
        error_response, 'ApplyGuardrail'
    )

    with pytest.raises(ClientError) as exc_info:
        bedrock_guardrail.apply_guardrail("test content")
    
    assert exc_info.value.response['Error']['Message'] == 'AWS Error'

def test_custom_runtime_client(guardrail_config):
    """Test guardrail with custom runtime client"""
    mock_client = Mock()
    guardrail = BedrockGuardrail(
        guardrail_id=guardrail_config['guardrail_id'],
        guardrail_version=guardrail_config['guardrail_version'],
        runtime_client=mock_client
    )
    
    assert guardrail.runtime_client == mock_client

@pytest.mark.parametrize("source_type,content", [
    ("INPUT", "test input content"),
    ("OUTPUT", "test output content"),
])
def test_different_source_types(bedrock_guardrail, mock_boto3_client, source_type, content):
    """Test guardrail with different source types"""
    mock_boto3_client.apply_guardrail.return_value = {'action': 'PASS'}
    
    bedrock_guardrail.apply_guardrail(content, source=source_type)
    
    mock_boto3_client.apply_guardrail.assert_called_with(
        guardrailIdentifier=bedrock_guardrail.guardrail_id,
        guardrailVersion=bedrock_guardrail.guardrail_version,
        source=source_type,
        content=[{"text": {"text": content}}]
    )

@pytest.mark.parametrize("error_code,error_message", [
    ("InternalServerException", "Internal server error"),
    ("ValidationException", "Invalid parameter"),
])
def test_error_handling(bedrock_guardrail, mock_boto3_client, error_code, error_message):
    """Test different error scenarios"""
    error_response = {
        'Error': {
            'Code': error_code,
            'Message': error_message
        }
    }
    mock_boto3_client.apply_guardrail.side_effect = ClientError(
        error_response, 'ApplyGuardrail'
    )

    with pytest.raises(ClientError) as exc_info:
        bedrock_guardrail.apply_guardrail("test content")
    
    assert exc_info.value.response['Error']['Code'] == error_code
    assert exc_info.value.response['Error']['Message'] == error_message

def test_request_formatting(bedrock_guardrail, mock_boto3_client):
    """Test proper request formatting"""
    test_content = "test content"
    mock_boto3_client.apply_guardrail.return_value = {'action': 'PASS'}

    bedrock_guardrail.apply_guardrail(test_content)

    mock_boto3_client.apply_guardrail.assert_called_with(
        guardrailIdentifier=bedrock_guardrail.guardrail_id,
        guardrailVersion=bedrock_guardrail.guardrail_version,
        source='INPUT',
        content=[{"text": {"text": test_content}}]
    )

def test_none_client_creation(guardrail_config):
    """Test that guardrail creates its own client when None is provided"""
    with patch('boto3.client') as mock_boto3_client:
        guardrail = BedrockGuardrail(
            guardrail_id=guardrail_config['guardrail_id'],
            guardrail_version=guardrail_config['guardrail_version'],
            runtime_client=None
        )
        
        mock_boto3_client.assert_called_once_with(
            'bedrock-runtime',
            region_name=guardrail_config['region_name']
        )
