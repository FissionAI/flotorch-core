import json
import pytest
from unittest.mock import Mock
from flotorch_core.reader.json_reader import JSONReader
from flotorch_core.storage.storage import StorageProvider

# Generic test model class with valid attributes
class TestModel:
    def __init__(self, question: str, answer: str, **kwargs):  # <-- Add **kwargs
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("Invalid data type for question or answer.")
        self.question = question
        self.answer = answer

    def __eq__(self, other):
        return self.question == other.question and self.answer == other.answer


class TestJSONReader:
    @pytest.fixture
    def storage_mock(self):
        """Fixture to create a mock storage provider"""
        return Mock(spec=StorageProvider)

    @pytest.fixture
    def json_reader(self, storage_mock):
        """Fixture to create a JSONReader instance with mock storage"""
        return JSONReader(storage_mock)

    def test_init(self, storage_mock):
        """Test JSONReader initialization"""
        reader = JSONReader(storage_mock)
        assert reader.storage_provider == storage_mock

    def test_read_single_object(self, json_reader, storage_mock):
        """Test reading a single JSON object"""
        test_json = '{"question": "What is AI?", "answer": "Artificial Intelligence"}'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read("test.json")

        assert isinstance(result, dict)
        assert result == {"question": "What is AI?", "answer": "Artificial Intelligence"}

    def test_read_array(self, json_reader, storage_mock):
        """Test reading a JSON array"""
        test_json = '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read("test.json")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"question": "Q1", "answer": "A1"}
        assert result[1] == {"question": "Q2", "answer": "A2"}

    def test_read_chunked_data(self, json_reader, storage_mock):
        """Test reading JSON data that comes in chunks"""
        storage_mock.read.return_value = [
            b'{"question": "',
            b'What is AI?", "answer": "Artificial Intelligence"}'
        ]

        result = json_reader.read("test.json")

        assert isinstance(result, dict)
        assert result == {"question": "What is AI?", "answer": "Artificial Intelligence"}

    def test_read_invalid_json(self, json_reader, storage_mock):
        """Test reading invalid JSON data"""
        storage_mock.read.return_value = [b'{"invalid": json}']

        with pytest.raises(json.JSONDecodeError):
            json_reader.read("test.json")

    def test_read_empty_json(self, json_reader, storage_mock):
        """Test handling empty JSON data"""
        storage_mock.read.return_value = [b'']

        with pytest.raises(json.JSONDecodeError):
            json_reader.read("test.json")

    def test_read_with_unicode(self, json_reader, storage_mock):
        """Test reading JSON with unicode characters from multiple languages"""
        test_json = '''
        {
            "question": "¿Qué es la IA?", 
            "answer": "Inteligencia Artificial 🤖",
            "chinese": "人工智能",
            "arabic": "الذكاء الاصطناعي",
            "hindi": "आर्टिफिशियल इंटेलिजेंस",
            "japanese": "人工知能",
            "emoji": "🔥🚀"
        }
        '''
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read("test.json")

        assert result["question"] == "¿Qué es la IA?"
        assert result["answer"] == "Inteligencia Artificial 🤖"
        assert result["chinese"] == "人工智能"  # Chinese
        assert result["arabic"] == "الذكاء الاصطناعي"  # Arabic
        assert result["hindi"] == "आर्टिफिशियल इंटेलिजेंस"  # Hindi
        assert result["japanese"] == "人工知能"  # Japanese
        assert result["emoji"] == "🔥🚀"  # Emojis

    def test_storage_provider_error(self, json_reader, storage_mock):
        """Test handling storage provider errors"""
        storage_mock.read.side_effect = Exception("Storage error")

        with pytest.raises(Exception, match="Storage error"):
            json_reader.read("test.json")

    def test_read_as_model_single_object(self, json_reader, storage_mock):
        """Test converting single JSON object to TestModel"""
        test_json = '{"question": "What is AI?", "answer": "Artificial Intelligence"}'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TestModel)
        assert result[0].question == "What is AI?"
        assert result[0].answer == "Artificial Intelligence"

    def test_read_as_model_array(self, json_reader, storage_mock):
        """Test converting JSON array to TestModel objects"""
        test_json = '[{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, TestModel) for item in result)
        assert result[0].question == "Q1"
        assert result[0].answer == "A1"
        assert result[1].question == "Q2"
        assert result[1].answer == "A2"

    def test_read_as_model_missing_fields(self, json_reader, storage_mock):
        """Test handling JSON with missing required model attributes"""
        test_json = '{"question": "Missing answer"}'  # Missing 'answer'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        with pytest.raises(TypeError):
            json_reader.read_as_model("test.json", TestModel)

    def test_read_as_model_extra_fields(self, json_reader, storage_mock):
        """Test JSON with extra attributes (should ignore extras)"""
        test_json = '{"question": "What is AI?", "answer": "Artificial Intelligence", "extra": "ignored"}'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert result[0] == TestModel(question="What is AI?", answer="Artificial Intelligence")

    def test_read_as_model_wrong_data_type(self, json_reader, storage_mock):
        """Test JSON with wrong data types"""
        test_json = '{"question": 123, "answer": ["Invalid"]}'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        with pytest.raises(TypeError):
            json_reader.read_as_model("test.json", TestModel)

    def test_read_as_model_list_with_invalid_entry(self, json_reader, storage_mock):
        """Test JSON list with one valid and one invalid entry"""
        test_json = '[{"question": "Valid", "answer": "Entry"}, {"question": "Invalid"}]'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        with pytest.raises(TypeError):
            json_reader.read_as_model("test.json", TestModel)
