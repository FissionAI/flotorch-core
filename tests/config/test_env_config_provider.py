import os
import pytest
from flotorch_core.config.env_config_provider import EnvConfigProvider

@pytest.fixture
def env_config():
    return EnvConfigProvider()

def test_get_existing_env_variable(env_config):
    # Setup
    test_key = "TEST_VAR"
    test_value = "test_value"
    os.environ[test_key] = test_value
    
    # Execute
    result = env_config.get(test_key)
    
    # Assert
    assert result == test_value
    
    # Cleanup
    del os.environ[test_key]

def test_get_non_existing_env_variable_with_default(env_config):
    # Execute
    result = env_config.get("NON_EXISTING_VAR", default="default_value")
    
    # Assert
    assert result == "default_value"

def test_get_non_existing_env_variable_without_default(env_config):
    # Execute
    result = env_config.get("NON_EXISTING_VAR")
    
    # Assert
    assert result is None
