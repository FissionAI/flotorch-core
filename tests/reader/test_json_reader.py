import json
import pytest
from unittest.mock import Mock
from flotorch_core.reader.json_reader import JSONReader
from flotorch_core.storage.storage import StorageProvider

# Test model class that accepts any attributes
class TestModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, TestModel):
            return False
        return self.__dict__ == other.__dict__

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
        test_json = '{"field1": "value1", "field2": "value2"}'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read("test.json")

        assert isinstance(result, dict)
        assert result == {"field1": "value1", "field2": "value2"}

    def test_read_array(self, json_reader, storage_mock):
        """Test reading a JSON array"""
        test_json = '[{"field1": "v1", "field2": "v2"}, {"field1": "v3", "field2": "v4"}]'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read("test.json")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"field1": "v1", "field2": "v2"}
        assert result[1] == {"field1": "v3", "field2": "v4"}

    def test_read_chunked_data(self, json_reader, storage_mock):
        """Test reading JSON data that comes in chunks"""
        storage_mock.read.return_value = [
            b'{"field1": "',
            b'partial value", "field2": "another value"}'
        ]

        result = json_reader.read("test.json")

        assert isinstance(result, dict)
        assert result == {"field1": "partial value", "field2": "another value"}

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
            "spanish": "¿Qué es la IA?", 
            "spanish_answer": "Inteligencia Artificial 🤖",
            "chinese": "人工智能",
            "arabic": "الذكاء الاصطناعي",
            "hindi": "आर्टिफिशियल इंटेलिजेंस",
            "japanese": "人工知能",
            "emoji": "🔥🚀"
        }
        '''
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read("test.json")

        assert result["spanish"] == "¿Qué es la IA?"
        assert result["spanish_answer"] == "Inteligencia Artificial 🤖"
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
        test_json = '{"field1": "value1", "field2": "value2"}'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TestModel)
        assert result[0].field1 == "value1"
        assert result[0].field2 == "value2"

    def test_read_as_model_array(self, json_reader, storage_mock):
        """Test converting JSON array to TestModel objects"""
        test_json = '[{"field1": "v1", "field2": "v2"}, {"field1": "v3", "field2": "v4"}]'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, TestModel) for item in result)
        assert result[0].field1 == "v1"
        assert result[0].field2 == "v2"
        assert result[1].field1 == "v3"
        assert result[1].field2 == "v4"

    def test_read_as_model_different_field_names(self, json_reader, storage_mock):
        """Test handling JSON with different field names across objects"""
        test_json = '[{"field1": "v1", "field2": "v2"}, {"field3": "v3", "field4": "v4"}]'
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].field1 == "v1"
        assert result[0].field2 == "v2"
        assert result[1].field3 == "v3"
        assert result[1].field4 == "v4"

    def test_read_as_model_mixed_types(self, json_reader, storage_mock):
        """Test handling JSON with various data types"""
        test_json = '''
        {
            "string_field": "text value",
            "number_field": 42,
            "float_field": 3.14,
            "boolean_field": true,
            "null_field": null,
            "array_field": [1, 2, 3],
            "object_field": {"nested": "value"}
        }
        '''
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].string_field == "text value"
        assert result[0].number_field == 42
        assert result[0].float_field == 3.14
        assert result[0].boolean_field is True
        assert result[0].null_field is None
        assert result[0].array_field == [1, 2, 3]
        assert result[0].object_field == {"nested": "value"}

    def test_read_as_model_custom_validation(self, json_reader, storage_mock):
        """Test with a model that has custom validation"""
        class ModelWithValidation:
            def __init__(self, **kwargs):
                # Only require at least one field to be present
                if not kwargs:
                    raise ValueError("At least one field is required")
                for key, value in kwargs.items():
                    setattr(self, key, value)

        test_json = '[{"valid": "data"}, {}]'  # Second object is empty
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        with pytest.raises(ValueError):
            json_reader.read_as_model("test.json", ModelWithValidation)

    def test_read_as_model_with_nested_objects(self, json_reader, storage_mock):
        """Test handling JSON with nested objects"""
        test_json = '''
        {
            "main_field": "value",
            "nested": {
                "sub_field1": "nested value",
                "sub_field2": 42
            }
        }
        '''
        storage_mock.read.return_value = [test_json.encode('utf-8')]

        result = json_reader.read_as_model("test.json", TestModel)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].main_field == "value"
        assert result[0].nested == {"sub_field1": "nested value", "sub_field2": 42}