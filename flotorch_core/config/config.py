from .config_provider import ConfigProvider

class Config:
    """
    Main configuration class that uses a configuration provider.
    """

    def __init__(self, provider: ConfigProvider):
        """
        Initializes the Config class with the specified provider.
        Args:
            provider (ConfigProvider): The configuration provider.
        """
        self.provider = provider

    def get_region(self) -> str:
        """
        Retrieves the AWS region from the configuration provider.
        Returns:
            str: The AWS region.
        Raises:
            ValueError: If the region is not set.
        """
        region = self.provider.get("AWS_REGION", "us-east-1")
        if not region:
            raise ValueError("AWS region is not set. Value not presnt in configuration")
        return region
    
    def get_opensearch_host(self) -> str:
        """
        Retrieves the OpenSearch host from the configuration provider.
        Returns:
            str: The OpenSearch host.
        Raises:
            ValueError: If the OpenSearch host is not set.
        """
        open_search_host = self.provider.get("OPENSEARCH_HOST", "localhost")
        if not open_search_host:
            raise ValueError("OpenSearch host is not set. Value not presnt in configuration")
        return open_search_host

    def get_opensearch_port(self) -> int:
        """
        Retrieves the OpenSearch port from the configuration provider.
        Returns:
            int: The OpenSearch port.
        Raises:
            ValueError: If the OpenSearch port is not set.
        """
        open_search_port = int(self.provider.get("OPENSEARCH_PORT", 443))
        if not open_search_port:
            raise ValueError("OpenSearch port is not set. Value not presnt in configuration")
        return open_search_port
    
    def get_opensearch_username(self) -> str:
        """
        Retrieves the OpenSearch username from the configuration provider.
        Returns:
            str: The OpenSearch username.
        Raises:
            ValueError: If the OpenSearch username is not set.
        """
        open_search_username = self.provider.get("OPENSEARCH_USERNAME")
        if not open_search_username:
            raise ValueError("OpenSearch username is not set. Value not presnt in configuration")
        return open_search_username

    def get_opensearch_password(self) -> str:
        """
        Retrieves the OpenSearch password from the configuration provider.
        Returns:
            str: The OpenSearch password.
        Raises:
            ValueError: If the OpenSearch password is not set.
        """
        open_search_password = self.provider.get("OPENSEARCH_PASSWORD")
        if not open_search_password:
            raise ValueError("OpenSearch password is not set. Value not presnt in configuration")
        return open_search_password

    def get_opensearch_index(self) -> str:
        """
        Retrieves the OpenSearch index name from the configuration provider.
        Returns:
            str: The OpenSearch index name.
        Raises:
            ValueError: If the OpenSearch index name is not set.
        """
        open_search_index = self.provider.get("OPENSEARCH_INDEX")
        if not open_search_index:
            raise ValueError("OpenSearch index is not set. Value not presnt in configuration")
        return open_search_index

    def get_task_token(self) -> str:
        """
        Retrieves the task token from the configuration provider.
        Returns:
            str: The task token.
        Raises:
            ValueError: If the task token is not set.
        """
        task_token = self.provider.get("TASK_TOKEN")
        if not task_token:
            raise ValueError("task token is not set. Value not presnt in configuration")
        return task_token

    def get_fargate_input_data(self) -> str:
        """
        Retrieves the input data for fargate handlers from the configuration provider.
        Returns:
            str: The input data for fargate handlers.
        Raises:
            ValueError: If the input data is not set.
        """
        input_data = self.provider.get("INPUT_DATA", {})
        if not input_data:
            raise ValueError("input data is not set. Value not present in configuration")
        return input_data
    
    def get_experiment_table_name(self) -> str:
        """
        Retrieves the experiment table name from the configuration provider.
        Returns:
            str: The experiment table name.
        Raises:
            ValueError: If the experiment table name is not set.
        """
        experiment_table_name = self.provider.get("experiment_table")
        if not experiment_table_name:
            raise ValueError("experiment table name is not set. Value not present in configuration")
        return experiment_table_name
    
    def get_execution_table_name(self) -> str:
        """
        Retrieves the execution table name from the configuration provider.
        Returns:
            str: The execution table name.
        Raises:
            ValueError: If the execution table name is not set.
        """
        execution_table_name = self.provider.get("execution_table")
        if not execution_table_name:
            raise ValueError("execution table name is not set. Value not present in configuration")
        return execution_table_name
    
    def get_experiment_question_metrics_table(self) -> str:
        """
        Retrieves the experiment question metrics table name from the configuration provider.
        Returns:
            str: The experiment question metrics table name.
        Raises:
            ValueError: If the experiment question metrics table name is not set.
        """
        experiment_question_metrics_table = self.provider.get("experiment_question_metrics_table")
        if not experiment_question_metrics_table:
            raise ValueError("experiment question metrics table name is not set. Value not present in configuration")
        return experiment_question_metrics_table
    
    def get_sagemaker_arn_role(self) -> str:
        """
        Retrieves the SageMaker ARN role from the configuration provider.
        Returns:
            str: The SageMaker ARN role.
        Raises:
            ValueError: If the SageMaker ARN role is not set.
        """
        sagemaker_arn_role = self.provider.get("sagemaker_role_arn")
        if not sagemaker_arn_role:
            raise ValueError("sagemaker arn role is not set. Value not present in configuration")
        return sagemaker_arn_role
    
    def get_experimentid_index(self) -> str:
        """
        Retrieves the experiment ID index name from the configuration provider.
        Returns:
            str: The experiment ID index name.
        Raises:
            ValueError: If the experiment ID index name is not set.
        """
        experiment_id_index = self.provider.get("experiment_question_metrics_experimentid_index")
        if not experiment_id_index:
            raise ValueError("experiment id index is not set. Value not present in configuration")
        return experiment_id_index
    
    def get_postgres_db(self) -> str:
        """
        Retrieves the Postgres database name from the configuration provider.
        """
        postgres_db = self.provider.get("postgres_db_name")
        if not postgres_db:
            raise ValueError("Postgres database is not set. Value not present in configuration")
        return postgres_db
    
    def get_postgres_user(self) -> str:
        """
        Retrieves the Postgres user from the configuration provider.
        """
        postgres_user = self.provider.get("postgres_user")
        if not postgres_user:
            raise ValueError("Postgres user is not set. Value not present in configuration")
        return postgres_user
    
    def get_postgres_password(self) -> str:
        """
        Retrieves the Postgres password from the configuration provider.
        """
        postgres_password = self.provider.get("postgres_password")
        if not postgres_password:
            raise ValueError("Postgres password is not set. Value not present in configuration")
        return postgres_password
    
    def get_postgres_host(self) -> str:
        """
        Retrieves the Postgres host from the configuration provider.
        """
        postgres_host = self.provider.get("postgres_host")
        if not postgres_host:
            raise ValueError("Postgres host is not set. Value not present in configuration")
        return postgres_host

    def get_postgres_port(self) -> int:
        """
        Retrieves the Postgres port from the configuration provider.
        """
        postgres_port = int(self.provider.get("postgres_port", 5432))
        if not postgres_port:
            raise ValueError("Postgres port is not set. Value not present in configuration")
        return postgres_port
    
    def get_db_type(self) -> str:
        """
        Retrieves the database type from the configuration provider.
        """
        db_type = self.provider.get("db_type")
        if not db_type:
            raise ValueError("db type is not set. Value not present in configuration")
        return db_type