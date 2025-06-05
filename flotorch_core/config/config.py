from .config_provider import ConfigProvider
from typing import Any

class Config:
    def __init__(self, provider: ConfigProvider):
        """Initialize configuration with a given provider."""
        self.provider = provider

    def _get_required(self, key: str, default: Any, msg: str) -> Any:
        """Internal helper to fetch a required config value or raise error."""
        value = self.provider.get(key, default)
        if value is None:
            raise ValueError(msg)
        return value

    def get_region(self, default: Any = None) -> str:
        """Return the AWS region to use."""
        return self._get_required("AWS_REGION", default, "AWS region is not set. Value not present in configuration")

    def get_opensearch_host(self, default: Any = None) -> str:
        """Return the OpenSearch host address."""
        return self._get_required("OPENSEARCH_HOST", default, "OpenSearch host is not set. Value not present in configuration")

    def get_opensearch_port(self, default: Any = 443) -> int:
        """Return the OpenSearch service port."""
        return int(self._get_required("OPENSEARCH_PORT", default, "OpenSearch port is not set. Value not present in configuration"))

    def get_opensearch_username(self, default: Any = None) -> str:
        """Return the OpenSearch username."""
        return self._get_required("OPENSEARCH_USERNAME", default, "OpenSearch username is not set. Value not present in configuration")

    def get_opensearch_password(self, default: Any = None) -> str:
        """Return the OpenSearch password."""
        return self._get_required("OPENSEARCH_PASSWORD", default, "OpenSearch password is not set. Value not present in configuration")

    def get_opensearch_index(self, default: Any = None) -> str:
        """Return the OpenSearch index name."""
        return self._get_required("OPENSEARCH_INDEX", default, "OpenSearch index is not set. Value not present in configuration")

    def get_task_token(self, default: Any = None) -> str:
        """Return the task token for the operation."""
        return self._get_required("TASK_TOKEN", default, "Task token is not set. Value not present in configuration")

    def get_fargate_input_data(self, default: Any = None) -> str:
        """Return the input data used for Fargate handler."""
        return self._get_required("INPUT_DATA", default, "Input data is not set. Value not present in configuration")

    def get_experiment_table_name(self, default: Any = None) -> str:
        """Return the name of the experiment table."""
        return self._get_required("experiment_table", default, "Experiment table name is not set. Value not present in configuration")

    def get_execution_table_name(self, default: Any = None) -> str:
        """Return the name of the execution table."""
        return self._get_required("execution_table", default, "Execution table name is not set. Value not present in configuration")

    def get_experiment_question_metrics_table(self, default: Any = None) -> str:
        """Return the experiment question metrics table name."""
        return self._get_required("experiment_question_metrics_table", default, "Experiment question metrics table name is not set. Value not present in configuration")

    def get_execution_model_invocations_table(self, default: Any = None) -> str:
        """Return the execution model invocations table name."""
        return self._get_required("execution_model_invocations_table", default, "Execution model invocations table name is not set. Value not present in configuration")

    def get_sagemaker_arn_role(self, default: Any = None) -> str:
        """Return the ARN role for SageMaker execution."""
        return self._get_required("sagemaker_role_arn", default, "Sagemaker ARN role is not set. Value not present in configuration")

    def get_experimentid_index(self, default: Any = None) -> str:
        """Return the index name for experiment ID queries."""
        return self._get_required("experiment_question_metrics_experimentid_index", default, "Experiment ID index is not set. Value not present in configuration")

    def get_postgres_db(self, default: Any = None) -> str:
        """Return the name of the Postgres database."""
        return self._get_required("postgres_db_name", default, "Postgres database is not set. Value not present in configuration")

    def get_postgres_user(self, default: Any = None) -> str:
        """Return the Postgres username."""
        return self._get_required("postgres_user", default, "Postgres user is not set. Value not present in configuration")

    def get_postgres_password(self, default: Any = None) -> str:
        """Return the Postgres password."""
        return self._get_required("postgres_password", default, "Postgres password is not set. Value not present in configuration")

    def get_postgres_host(self, default: Any = None) -> str:
        """Return the Postgres host address."""
        return self._get_required("postgres_host", default, "Postgres host is not set. Value not present in configuration")

    def get_postgres_port(self, default: Any = 5432) -> int:
        """Return the port number for Postgres service."""
        return int(self._get_required("postgres_port", default, "Postgres port is not set. Value not present in configuration"))

    def get_db_type(self, default: Any = None) -> str:
        """Return the database type used in the system."""
        return self._get_required("db_type", default, "DB type is not set. Value not present in configuration")