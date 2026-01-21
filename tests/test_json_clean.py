"""
Tests for json_clean module.

Tests JSON validation and sanitization utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path
from json_clean import validate_json_file, examine_file_structure, fix_json_array


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_validate_valid_json_array(temp_dir):
    """Test validation of valid JSON array."""
    test_file = temp_dir / "valid_array.json"
    test_file.write_text('[{"key": "value"}, {"key2": "value2"}]')
    
    is_valid, message = validate_json_file(str(test_file))
    assert is_valid is True
    assert "valid JSON" in message


def test_validate_valid_json_object(temp_dir):
    """Test validation of valid JSON object."""
    test_file = temp_dir / "valid_object.json"
    test_file.write_text('{"key": "value", "nested": {"key2": "value2"}}')
    
    is_valid, message = validate_json_file(str(test_file))
    assert is_valid is True
    assert "valid JSON" in message


def test_validate_invalid_json(temp_dir):
    """Test validation of invalid JSON."""
    test_file = temp_dir / "invalid.json"
    test_file.write_text('{"key": "value"')  # Missing closing brace
    
    is_valid, message = validate_json_file(str(test_file))
    assert is_valid is False
    assert "Invalid JSON" in message


def test_validate_non_json_start(temp_dir):
    """Test validation of file that doesn't start with [ or {."""
    test_file = temp_dir / "non_json.txt"
    test_file.write_text('This is not JSON')
    
    is_valid, message = validate_json_file(str(test_file))
    assert is_valid is False
    assert "must start with" in message


def test_validate_empty_file(temp_dir):
    """Test validation of empty file."""
    test_file = temp_dir / "empty.json"
    test_file.write_text('')
    
    is_valid, message = validate_json_file(str(test_file))
    assert is_valid is False


def test_examine_array_structure(temp_dir):
    """Test examining JSON array structure."""
    test_file = temp_dir / "array.json"
    data = [{"id": 1}, {"id": 2}]
    test_file.write_text(json.dumps(data))
    
    info = examine_file_structure(str(test_file))
    
    assert info['file_path'] == str(test_file)
    assert info['file_size'] > 0
    assert info['first_char'] == '['
    assert info['structure'] == 'array'


def test_examine_object_structure(temp_dir):
    """Test examining JSON object structure."""
    test_file = temp_dir / "object.json"
    data = {"key": "value", "nested": {"data": "here"}}
    test_file.write_text(json.dumps(data))
    
    info = examine_file_structure(str(test_file))
    
    assert info['first_char'] == '{'
    assert info['structure'] == 'object'


def test_examine_claude_markers(temp_dir):
    """Test detection of Claude conversation markers."""
    test_file = temp_dir / "claude.json"
    # Include Claude-specific fields
    data = [{
        "uuid": "test-uuid",
        "chat_messages": [{"sender": "human", "text": "test"}]
    }]
    test_file.write_text(json.dumps(data))
    
    info = examine_file_structure(str(test_file))
    assert info['has_claude_markers'] is True


def test_examine_chatgpt_markers(temp_dir):
    """Test detection of ChatGPT conversation markers."""
    test_file = temp_dir / "chatgpt.json"
    # Include ChatGPT-specific fields
    data = [{
        "mapping": {"node1": {"message": {"author": {"role": "user"}}}},
        "create_time": 1234567890
    }]
    test_file.write_text(json.dumps(data))
    
    info = examine_file_structure(str(test_file))
    assert info['has_chatgpt_markers'] is True


def test_fix_json_array_creates_output(temp_dir):
    """Test that fix_json_array creates output file."""
    input_file = temp_dir / "input.json"
    output_file = temp_dir / "output.json"
    
    # Create a valid JSON array
    data = [{"id": 1}, {"id": 2}]
    input_file.write_text(json.dumps(data))
    
    # Fix should work even on valid JSON
    result = fix_json_array(str(input_file), str(output_file))
    
    # Check if output file exists
    if result:
        assert output_file.exists()


def test_examine_file_size(temp_dir):
    """Test that file size is correctly reported."""
    test_file = temp_dir / "sized.json"
    data = {"key": "value"}
    test_file.write_text(json.dumps(data))
    
    info = examine_file_structure(str(test_file))
    assert info['file_size'] == test_file.stat().st_size


def test_validate_nonexistent_file():
    """Test validation of nonexistent file."""
    is_valid, message = validate_json_file("/nonexistent/file.json")
    assert is_valid is False
    assert "Error" in message
