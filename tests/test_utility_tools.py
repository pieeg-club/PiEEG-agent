"""Tests for UtilityTools — file I/O, notebooks, and tool discovery."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pieeg_agent.agent import UtilityTools


@pytest.fixture
def utility():
    """Create a UtilityTools instance without other toolsets."""
    return UtilityTools()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_list_tools(utility):
    """Test list_tools enumerates all utility tools."""
    # Default: concise output
    result = utility.call("list_tools")
    assert result["count"] > 0
    assert "by_toolset" in result
    assert "toolsets" in result
    assert "utility" in result["toolsets"]
    
    # Check tools are grouped by toolset
    utility_tools = result["by_toolset"]["utility"]
    tool_names = [t["name"] for t in utility_tools]
    assert "list_tools" in tool_names
    assert "read_file" in tool_names
    
    # Concise mode should not include full schemas
    assert "parameters" not in utility_tools[0]
    
    # Verbose mode
    verbose_result = utility.call("list_tools", {"verbose": True})
    verbose_tools = verbose_result["by_toolset"]["utility"]
    assert "parameters" in verbose_tools[0]
    assert "required" in verbose_tools[0]
    
    # Filter by toolset
    filtered = utility.call("list_tools", {"toolset": "utility"})
    assert "utility" in filtered["by_toolset"]
    assert len(filtered["toolsets"]) == 1


def test_write_and_read_file(utility, temp_dir):
    """Test writing and reading a text file."""
    test_path = temp_dir / "test.txt"
    content = "Hello, PiEEG!\nLine 2\nLine 3"
    
    # Write file
    write_result = utility.call("write_file", {
        "path": str(test_path),
        "content": content,
    })
    assert write_result["status"] == "written"
    assert write_result["path"] == str(test_path)
    assert write_result["size"] > 0
    
    # Read file
    read_result = utility.call("read_file", {"path": str(test_path)})
    assert read_result["content"] == content
    assert read_result["size"] == write_result["size"]


def test_write_file_creates_parent_dirs(utility, temp_dir):
    """Test that write_file creates parent directories."""
    nested_path = temp_dir / "a" / "b" / "c" / "file.txt"
    
    result = utility.call("write_file", {
        "path": str(nested_path),
        "content": "nested",
    })
    assert result["status"] == "written"
    assert nested_path.exists()


def test_read_nonexistent_file(utility):
    """Test reading a file that doesn't exist."""
    result = utility.call("read_file", {"path": "/nonexistent/path/file.txt"})
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_list_directory(utility, temp_dir):
    """Test listing directory contents."""
    # Create some test files
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.txt").write_text("content2")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file3.txt").write_text("content3")
    
    # Non-recursive list
    result = utility.call("list_directory", {"path": str(temp_dir)})
    assert result["count"] == 3  # file1, file2, subdir
    assert "entries" in result
    
    names = [e["name"] for e in result["entries"]]
    assert "file1.txt" in names
    assert "file2.txt" in names
    assert "subdir" in names
    
    # Check types
    types = {e["name"]: e["type"] for e in result["entries"]}
    assert types["file1.txt"] == "file"
    assert types["subdir"] == "directory"


def test_list_directory_recursive(utility, temp_dir):
    """Test recursive directory listing."""
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file2.txt").write_text("content2")
    
    result = utility.call("list_directory", {
        "path": str(temp_dir),
        "recursive": True,
    })
    assert result["count"] >= 3  # file1, subdir, subdir/file2
    
    paths = [e["path"] for e in result["entries"]]
    assert "file1.txt" in paths
    assert str(Path("subdir") / "file2.txt") in paths or "subdir\\file2.txt" in paths


def test_create_notebook(utility, temp_dir):
    """Test creating a Jupyter notebook."""
    nb_path = temp_dir / "test.ipynb"
    
    cells = [
        {"type": "markdown", "source": "# Test Notebook\nThis is a test."},
        {"type": "code", "source": "x = 1 + 1\nprint(x)"},
        {"type": "code", "source": "y = x * 2\ny"},
    ]
    
    result = utility.call("create_notebook", {
        "path": str(nb_path),
        "cells": cells,
    })
    assert result["status"] == "created"
    assert result["cells"] == 3  # No header added when no metadata
    assert nb_path.exists()
    
    # Verify it's valid JSON
    with open(nb_path) as f:
        nb_data = json.load(f)
    assert "cells" in nb_data
    assert len(nb_data["cells"]) == 3


def test_create_notebook_with_metadata(temp_dir):
    """Test creating a notebook with EEG session metadata."""
    metadata = {
        "stream_name": "MockEEG",
        "channels": {
            "count": 8,
            "labels": ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
        },
        "sample_rate": 250.0,
        "mains_hz": 60,
        "provider": "anthropic:claude-3-5-sonnet-20241022",
    }
    utility = UtilityTools(session_metadata=metadata)
    
    nb_path = temp_dir / "test_meta.ipynb"
    cells = [
        {"type": "markdown", "source": "# Analysis"},
        {"type": "code", "source": "print('hello')"},
    ]
    
    result = utility.call("create_notebook", {
        "path": str(nb_path),
        "cells": cells,
    })
    assert result["status"] == "created"
    assert result["cells"] == 3  # Header + 2 user cells
    assert "metadata" in result
    assert "pieeg" in result["metadata"]
    
    # Verify metadata is embedded
    with open(nb_path) as f:
        nb_data = json.load(f)
    assert "pieeg" in nb_data["metadata"]
    assert nb_data["metadata"]["pieeg"]["stream_name"] == "MockEEG"
    assert nb_data["metadata"]["pieeg"]["sample_rate"] == 250.0
    
    # Check header cell contains session info
    header_cell = nb_data["cells"][0]
    assert header_cell["cell_type"] == "markdown"
    # Source might be a string or list of strings
    source = header_cell["source"]
    if isinstance(source, list):
        source = "".join(source)
    assert "PiEEG Analysis Notebook" in source
    assert "MockEEG" in source
    assert "250 Hz" in source


def test_read_notebook(utility, temp_dir):
    """Test reading a notebook structure."""
    nb_path = temp_dir / "test.ipynb"
    
    cells = [
        {"type": "markdown", "source": "# Header"},
        {"type": "code", "source": "print('hello')"},
    ]
    
    # Create notebook
    utility.call("create_notebook", {"path": str(nb_path), "cells": cells})
    
    # Read it back
    result = utility.call("read_notebook", {"path": str(nb_path)})
    assert "cells" in result
    assert len(result["cells"]) == 2
    assert result["cells"][0]["type"] == "markdown"
    assert result["cells"][0]["source"] == "# Header"
    assert result["cells"][1]["type"] == "code"
    assert result["cells"][1]["source"] == "print('hello')"


def test_run_notebook(utility, temp_dir):
    """Test executing a notebook and capturing outputs."""
    nb_path = temp_dir / "test.ipynb"
    
    cells = [
        {"type": "code", "source": "x = 2 + 2\nprint(f'x = {x}')"},
        {"type": "code", "source": "y = x * 3\ny"},
    ]
    
    # Create notebook
    utility.call("create_notebook", {"path": str(nb_path), "cells": cells})
    
    # Execute it
    result = utility.call("run_notebook", {"path": str(nb_path)})
    
    # Skip if no kernel available (common in test environments)
    if "error" in result:
        pytest.skip(f"Jupyter kernel not available: {result['error']}")
    
    assert result["status"] == "executed"
    assert "outputs" in result
    assert len(result["outputs"]) == 2
    
    # Check first cell output (print statement)
    assert result["outputs"][0]["cell"] == 0
    assert len(result["outputs"][0]["outputs"]) > 0
    assert result["outputs"][0]["outputs"][0]["type"] == "stream"
    assert "x = 4" in result["outputs"][0]["outputs"][0]["text"]
    
    # Check second cell output (expression result)
    assert result["outputs"][1]["cell"] == 1
    assert len(result["outputs"][1]["outputs"]) > 0


def test_create_notebook_with_invalid_cell_type(utility, temp_dir):
    """Test that invalid cell types are rejected."""
    nb_path = temp_dir / "test.ipynb"
    
    cells = [
        {"type": "invalid_type", "source": "content"},
    ]
    
    result = utility.call("create_notebook", {
        "path": str(nb_path),
        "cells": cells,
    })
    assert "error" in result
    assert "invalid cell type" in result["error"].lower()


def test_unknown_tool(utility):
    """Test calling an unknown tool returns an error."""
    result = utility.call("nonexistent_tool")
    assert "error" in result
    assert "unknown tool" in result["error"].lower()
    assert "available" in result


def test_specs_and_names(utility):
    """Test that specs and names return consistent results."""
    specs = utility.specs()
    names = utility.names()
    
    assert len(specs) == len(names)
    spec_names = [s.name for s in specs]
    assert set(spec_names) == set(names)
    
    # Verify all expected tools are present
    expected_tools = {
        "list_tools", "read_file", "write_file", "read_image",
        "list_directory", "create_notebook", "run_notebook", "read_notebook",
    }
    assert expected_tools.issubset(set(names))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
