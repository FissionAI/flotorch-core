import pytest
from unittest.mock import Mock, patch
from flotorch_core.storage.db.vector.guardrails_vector_storage import GuardRailsVectorStorage
from flotorch_core.storage.db.vector.vector_storage import VectorStorage, VectorStorageSearchResponse
from flotorch_core.guardrails.guardrails import BaseGuardRail
from flotorch_core.chunking.chunking import Chunk

class TestGuardRailsVectorStorage:
    @pytest.fixture
    def mock_vector_storage(self):
        """Create a mock VectorStorage"""
        mock = Mock(spec=VectorStorage)
        mock.search.return_value = VectorStorageSearchResponse(
            status=True,
            result=[],
            metadata={'embedding_metadata': {}}
        )
        return mock

    @pytest.fixture
    def mock_guardrail(self):
        """Create a mock BaseGuardRail with proper return values"""
        mock = Mock(spec=BaseGuardRail)
        # Configure the mock to return a proper guardrail response
        mock.apply_guardrail.return_value = {
            'action': 'PROCEED',
            'outputs': [{'text': 'Safe content'}],
            'assessments': []
        }
        return mock

    @pytest.fixture
    def storage(self, mock_vector_storage, mock_guardrail):
        """Create a GuardRailsVectorStorage instance for testing"""
        return GuardRailsVectorStorage(
            vectorStorage=mock_vector_storage,
            base_guardrail=mock_guardrail,
            apply_prompt=True,
            apply_context=True
        )

    def test_search_prompt_guardrail_intervened(self, storage, mock_guardrail):
        """Test search when prompt guardrail intervenes"""
        # Configure mock guardrail to return an intervention response
        mock_guardrail.apply_guardrail.return_value = {
            'action': 'GUARDRAIL_INTERVENED',
            'outputs': [{'text': 'Unsafe content detected'}],
            'assessments': ['unsafe content']
        }

        test_chunk = Chunk(data="test query")
        response = storage.search(test_chunk, knn=5)

        # Verify response
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is False
        assert response.metadata['guardrail_output'] == 'Unsafe content detected'
        assert response.metadata['guardrail_input_assessment'] == ['unsafe content']
        assert response.metadata['block_level'] == 'INPUT'
        assert response.metadata['guardrail_blocked'] is True

    def test_search_context_guardrail_intervened(self, storage, mock_guardrail, mock_vector_storage):
        """Test search when context guardrail intervenes"""
        # Configure mock guardrail to proceed for prompt but intervene for context
        mock_guardrail.apply_guardrail.side_effect = [
            {
                'action': 'PROCEED',
                'outputs': [{'text': 'Safe prompt'}],
                'assessments': []
            },
            {
                'action': 'GUARDRAIL_INTERVENED',
                'outputs': [{'text': 'Unsafe context detected'}],
                'assessments': ['unsafe context']
            }
        ]

        # Configure mock vector storage to return some results
        mock_result = Mock()
        mock_result.text = "test result"
        mock_vector_storage.search.return_value = VectorStorageSearchResponse(
            status=True,
            result=[mock_result],
            metadata={'embedding_metadata': {}}
        )

        test_chunk = Chunk(data="test query")
        response = storage.search(test_chunk, knn=5)

        # Verify response
        assert isinstance(response, VectorStorageSearchResponse)
        assert response.status is False
        assert response.metadata['guardrail_output'] == 'Unsafe context detected'
        assert response.metadata['guardrail_context_assessment'] == ['unsafe context']
        assert response.metadata['block_level'] == 'CONTEXT'
        assert response.metadata['guardrail_blocked'] is True

    def test_search_no_intervention(self, storage, mock_guardrail):
        """Test search with no guardrail intervention"""
        # Configure mock guardrail to proceed
        mock_guardrail.apply_guardrail.return_value = {
            'action': 'PROCEED',
            'outputs': [{'text': 'Safe content'}],
            'assessments': []
        }

        test_chunk = Chunk(data="test query")
        response = storage.search(test_chunk, knn=5)

        # Verify the search proceeded normally
        assert response.status is True
        assert isinstance(response, VectorStorageSearchResponse)

    def test_search_with_prompt_disabled(self, storage):
        """Test search with prompt checking disabled"""
        storage.apply_prompt = False
        test_chunk = Chunk(data="test query")
        
        response = storage.search(test_chunk, knn=5)
        
        # Verify search was called without prompt check
        storage.vectorStorage.search.assert_called_once_with(test_chunk, 5, False)
        assert isinstance(response, VectorStorageSearchResponse)
