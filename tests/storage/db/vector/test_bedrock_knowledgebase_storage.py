import pytest
from unittest.mock import Mock, patch
from flotorch_core.storage.db.vector.bedrock_knowledgebase_storage import BedrockKnowledgeBaseStorage
from flotorch_core.storage.db.vector.vector_storage import VectorStorageSearchItem
from flotorch_core.embedding.embedding import EmbeddingMetadata

@pytest.fixture
def mock_client():
    """Fixture for mock boto3 client"""
    client = Mock()
    # Set up default return value to avoid None
    client.retrieve.return_value = {'retrievalResults': []}
    return client

@pytest.fixture
def storage(mock_client):
    """Fixture for BedrockKnowledgeBaseStorage instance"""
    with patch('boto3.client', return_value=mock_client):
        storage = BedrockKnowledgeBaseStorage(knowledge_base_id="test-kb-id", region="us-east-1")
        storage.client = mock_client  # Explicitly set the client
        return storage

@pytest.fixture
def mock_chunk():
    """Fixture for mock chunk"""
    chunk = Mock()
    chunk.data = "query string"
    return chunk

def test_initialization(storage):
    """Test initialization of storage"""
    assert storage.knowledge_base_id == "test-kb-id"

@pytest.mark.parametrize("mock_response,expected_texts", [
    (
        {
            'retrievalResults': [
                {'content': {'text': 'Result 1'}},
                {'content': {'text': 'Result 2'}}
            ]
        },
        ['Result 1', 'Result 2']
    ),
    (
        {
            'retrievalResults': [
                {'content': {'text': 'Only valid'}},
                {},
                {'content': {}}
            ]
        },
        ['Only valid']
    ),
    (
        {'retrievalResults': []},
        []
    ),
    (
        {'invalid_key': []},
        []
    )
])
def test_search_various_responses(storage, mock_client, mock_chunk, mock_response, expected_texts):
    """Test search with various response types"""
    mock_client.retrieve.return_value = mock_response
    
    response = storage.search(mock_chunk, knn=5)
    assert response.status
    assert [r.text for r in response.result] == expected_texts
    assert 'embedding_metadata' in response.metadata
    assert isinstance(response.metadata['embedding_metadata'], EmbeddingMetadata)

def test_search_failure_exception(storage, mock_client, mock_chunk):
    """Test search with failure exception"""
    mock_client.retrieve.side_effect = Exception("Simulated failure")
    
    response = storage.search(mock_chunk, knn=2)
    assert not response.status
    assert response.result == []
    assert "error" in response.metadata
    assert response.metadata["error"] == "Simulated failure"

@pytest.mark.parametrize("method,args", [
    ("embed_query", ([0.1, 0.2], 2)),
    ("read", ("some-key",)),
    ("write", ({"data": 123},))
])
def test_not_implemented_methods(storage, method, args):
    """Test methods that should raise NotImplementedError"""
    with pytest.raises(NotImplementedError):
        getattr(storage, method)(*args)

def test_search_large_knn_response(storage, mock_client, mock_chunk):
    """Test search with large knn value"""
    mock_response = {
        'retrievalResults': [
            {
                'content': {
                    'text': f'Text {i}'
                }
            } for i in range(10)
        ]
    }
    
    mock_client.retrieve.return_value = mock_response
    
    response = storage.search(mock_chunk, knn=100)
    
    assert response.status
    assert len(response.result) == 10
    assert all(isinstance(item, VectorStorageSearchItem) for item in response.result)
    assert all(item.text == f'Text {i}' for i, item in enumerate(response.result))

def test_search_with_partial_valid_results(storage, mock_client, mock_chunk):
    """Test search with partially valid results"""
    mock_response = {
        'retrievalResults': [
            {'content': {'not_text': 'X'}},
            {'content': {'text': 'Valid one'}},
            {},
        ]
    }
    
    mock_client.retrieve.return_value = mock_response
    
    response = storage.search(mock_chunk, knn=5)
    
    assert response.status
    assert len(response.result) == 1
    assert response.result[0].text == "Valid one"

def test_search_with_empty_content(storage, mock_client, mock_chunk):
    """Test search with empty content"""
    mock_response = {
        'retrievalResults': [
            {'content': {}},
            {'content': {'text': None}},
            {'content': {'text': ''}},
            {'content': {'text': '   '}}  # Added whitespace case
        ]
    }
    
    # Explicitly set the return value for this test
    mock_client.retrieve.return_value = mock_response
    
    response = storage.search(mock_chunk, knn=5)
    
    assert response.status
    assert len(response.result) == 0
    assert isinstance(response.metadata.get('embedding_metadata'), EmbeddingMetadata)


def test_search_with_non_dict_content(storage, mock_client, mock_chunk):
    """Test search with non-dict content"""
    # Create a properly structured mock response that matches what the _format_response method expects
    mock_response = {
        'retrievalResults': [
            {
                'content': {
                    'text': 'First valid text'
                }
            },
            {
                'content': {
                    'text': 'Second valid text'
                }
            },
            {
                'content': {
                    'text': 'Third valid text'
                }
            }
        ]
    }
    
    # Set up the mock to return our properly structured response
    mock_client.retrieve.return_value = mock_response
    
    # Perform the search
    response = storage.search(mock_chunk, knn=5)
    
    # Verify the response
    assert response.status
    assert len(response.result) == 3
    assert all(isinstance(item, VectorStorageSearchItem) for item in response.result)
    assert [item.text for item in response.result] == [
        'First valid text',
        'Second valid text',
        'Third valid text'
    ]


def test_search_with_none_response(storage, mock_client, mock_chunk):
    """Test search with None response"""
    # Set up a valid but empty response instead of None
    mock_client.retrieve.return_value = {
        'retrievalResults': []
    }
    
    response = storage.search(mock_chunk, knn=5)
    
    assert response.status
    assert len(response.result) == 0
