import pytest
from flotorch_core.storage.db.db_storage import DBStorage
from typing import List

# Create a concrete class for testing
class ConcreteStorageforTestDBStorage(DBStorage):
    def __init__(self):
        self.written_items = []
        self.read_items = {}

    def read(self, key) -> dict:
        return self.read_items.get(key)

    def write(self, item: dict):
        self.written_items.append(item)

class TestDBStorage:
    def test_cannot_instantiate_abstract_class(self):
        """Test that DBStorage cannot be instantiated directly"""
        with pytest.raises(TypeError) as exc_info:
            DBStorage()
        assert "Can't instantiate abstract class DBStorage" in str(exc_info.value)

    def test_must_implement_abstract_methods(self):
        """Test that subclasses must implement abstract methods"""
        class IncompleteStorage(DBStorage):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteStorage()
        assert "Can't instantiate abstract class IncompleteStorage" in str(exc_info.value)

    def test_bulk_write_implementation(self):
        """Test the concrete bulk_write implementation"""
        storage = ConcreteStorageforTestDBStorage()
        test_items = [
            {"id": "1", "data": "test1"},
            {"id": "2", "data": "test2"},
            {"id": "3", "data": "test3"}
        ]

        # Test bulk write
        storage.bulk_write(test_items)

        # Verify all items were written
        assert len(storage.written_items) == 3
        assert storage.written_items == test_items

    def test_bulk_write_empty_list(self):
        """Test bulk_write with empty list"""
        storage = ConcreteStorageforTestDBStorage()
        storage.bulk_write([])
        assert len(storage.written_items) == 0

    def test_bulk_write_single_item(self):
        """Test bulk_write with a single item"""
        storage = ConcreteStorageforTestDBStorage()
        test_item = {"id": "1", "data": "test1"}
        
        storage.bulk_write([test_item])
        
        assert len(storage.written_items) == 1
        assert storage.written_items[0] == test_item

    def test_abstract_methods_exist(self):
        """Test that abstract methods are properly defined"""
        # Get the list of abstract methods
        abstract_methods = DBStorage.__abstractmethods__
        
        # Verify required abstract methods
        assert 'read' in abstract_methods
        assert 'write' in abstract_methods

    def test_method_signatures(self):
        """Test that method signatures are correct"""
        storage = ConcreteStorageforTestDBStorage()
        
        # Test read method signature
        from inspect import signature
        read_sig = signature(storage.read)
        assert read_sig.return_annotation == dict  # Direct comparison with type
        
        # Test write method signature
        write_sig = signature(storage.write)
        assert 'item' in write_sig.parameters
        assert write_sig.parameters['item'].annotation == dict
        
        # Test bulk_write method signature
        bulk_write_sig = signature(storage.bulk_write)
        assert 'items' in bulk_write_sig.parameters
        # For List[dict], we need to check the _name attribute or use typing.get_origin/args
        from typing import get_origin, get_args
        assert get_origin(bulk_write_sig.parameters['items'].annotation) == list
        assert get_args(bulk_write_sig.parameters['items'].annotation)[0] == dict
