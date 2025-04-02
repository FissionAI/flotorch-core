import pytest
from flotorch_core.config.config import Config
from flotorch_core.config.config_provider import ConfigProvider

class MockConfigProvider(ConfigProvider):
    def __init__(self, config_dict=None):
        self.config_dict = config_dict or {}
    
    def get(self, key: str, default=None):
        return self.config_dict.get(key, default)

@pytest.fixture
def mock_provider():
    return MockConfigProvider({
        "AWS_REGION": "us-west-2",
        "OPENSEARCH_HOST": "test-host",
        "OPENSEARCH_PORT": "9200",
        "OPENSEARCH_USERNAME": "test-user",
        "OPENSEARCH_PASSWORD": "test-password",
        "OPENSEARCH_INDEX": "test-index",
        "TASK_TOKEN": "test-token",
        "INPUT_DATA": '{"key": "value"}',
        "experiment_table": "test-experiment-table",
        "experiment_question_metrics_table": "test-metrics-table",
        "sagemaker_role_arn": "test-role-arn"
    })

@pytest.fixture
def config(mock_provider):
    return Config(provider=mock_provider)

def test_config_initialization():
    provider = MockConfigProvider()
    config = Config(provider=provider)
    assert config.provider == provider

def test_get_region(config):
    assert config.get_region() == "us-west-2"

def test_get_region_default():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    assert config.get_region() == "us-east-1"  # Default value

def test_get_region_missing():
    provider = MockConfigProvider({"AWS_REGION": ""})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="AWS region is not set"):
        config.get_region()

def test_get_opensearch_host(config):
    assert config.get_opensearch_host() == "test-host"

def test_get_opensearch_host_default():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    assert config.get_opensearch_host() == "localhost"  # Default value

def test_get_opensearch_port(config):
    assert config.get_opensearch_port() == 9200

def test_get_opensearch_port_default():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    assert config.get_opensearch_port() == 443  # Default value

def test_get_opensearch_username(config):
    assert config.get_opensearch_username() == "test-user"

def test_get_opensearch_username_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="OpenSearch username is not set"):
        config.get_opensearch_username()

def test_get_opensearch_password(config):
    assert config.get_opensearch_password() == "test-password"

def test_get_opensearch_password_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="OpenSearch password is not set"):
        config.get_opensearch_password()

def test_get_opensearch_index(config):
    assert config.get_opensearch_index() == "test-index"

def test_get_opensearch_index_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="OpenSearch index is not set"):
        config.get_opensearch_index()

def test_get_task_token(config):
    assert config.get_task_token() == "test-token"

def test_get_task_token_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="task token is not set"):
        config.get_task_token()

def test_get_fargate_input_data(config):
    assert config.get_fargate_input_data() == '{"key": "value"}'

def test_get_fargate_input_data_default():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="input data is not set"):
        config.get_fargate_input_data()

def test_get_experiment_table_name(config):
    assert config.get_experiment_table_name() == "test-experiment-table"

def test_get_experiment_table_name_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="experiment table name is not set"):
        config.get_experiment_table_name()

def test_get_experiment_question_metrics_table(config):
    assert config.get_experiment_question_metrics_table() == "test-metrics-table"

def test_get_experiment_question_metrics_table_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="experiment question metrics table name is not set"):
        config.get_experiment_question_metrics_table()

def test_get_sagemaker_arn_role(config):
    assert config.get_sagemaker_arn_role() == "test-role-arn"

def test_get_sagemaker_arn_role_missing():
    provider = MockConfigProvider({})
    config = Config(provider=provider)
    with pytest.raises(ValueError, match="sagemaker arn role is not set"):
        config.get_sagemaker_arn_role()
