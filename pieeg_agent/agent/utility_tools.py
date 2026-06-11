"""Utility tools — file system, notebooks, and tool discovery.

This toolset provides general-purpose capabilities beyond neural sensing:

* ``list_tools``          — discover all currently available tools
* ``read_file``           — read text files from disk (code, logs, configs)
* ``read_image``          — read and describe image files (base64 data URI)
* ``write_file``          — write text content to a local file
* ``list_directory``      — list contents of a directory
* ``create_notebook``     — create a new Jupyter notebook with cells
* ``run_notebook``        — execute a Jupyter notebook and return outputs
* ``read_notebook``       — read notebook structure and cell outputs

These are designed for workflow automation, data analysis, and introspection.
Unlike the neural tools, these interact with the file system and may have
side effects (writing files, executing code). Use with care.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from .tools import Tool, Toolset, _spec

logger = logging.getLogger("pieeg.agent.utility")


class UtilityTools:
    """General-purpose utility toolset for file I/O, notebooks, and introspection."""

    def __init__(self, toolsets: list[Toolset] | None = None, session_metadata: dict | None = None):
        """Initialize utility tools.
        
        Args:
            toolsets: Optional list of other toolsets for tool discovery.
                     If provided, list_tools will enumerate all available tools.
            session_metadata: Optional EEG session metadata (channels, sample_rate, etc.)
                            to embed in created notebooks.
        """
        self._tools: dict[str, Tool] = {}
        self._other_toolsets = toolsets or []
        self._session_metadata = session_metadata or {}
        self._register_all()

    # ── registry surface (mirrors NeuralTools / ActuatorTools) ──────────
    def specs(self):
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "available": self.names()}
        try:
            return tool.handler(arguments or {})
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── registration ────────────────────────────────────────────────────
    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_all(self) -> None:
        # Tool discovery
        self._add(Tool(
            _spec(
                "list_tools",
                "List all currently available tools across all toolsets. "
                "Returns tool names and descriptions. Set verbose=true for "
                "full parameter schemas. Use this to discover what the agent can do.",
                {
                    "verbose": {
                        "type": "boolean",
                        "description": "Include full parameter schemas (default: false).",
                    },
                    "toolset": {
                        "type": "string",
                        "description": "Filter by toolset: neural, decode, documentation, actuator, utility.",
                    },
                },
            ),
            self._list_tools,
        ))

        # File system operations
        self._add(Tool(
            _spec(
                "read_file",
                "Read a text file from the local file system. Returns the file "
                "content as a string. Use for reading code, logs, configs, CSVs, etc.",
                {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding (default: utf-8).",
                    },
                },
                required=["path"],
            ),
            self._read_file,
        ))

        self._add(Tool(
            _spec(
                "read_image",
                "Read an image file and return it as a base64 data URI. "
                "Supports PNG, JPG, GIF, BMP, WebP. Useful for viewing plots, "
                "spectrograms, or user-generated images.",
                {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the image.",
                    },
                },
                required=["path"],
            ),
            self._read_image,
        ))

        self._add(Tool(
            _spec(
                "write_file",
                "Write text content to a local file. Creates parent directories "
                "if needed. Use for saving analysis results, generated configs, etc.",
                {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding (default: utf-8).",
                    },
                },
                required=["path", "content"],
            ),
            self._write_file,
        ))

        self._add(Tool(
            _spec(
                "list_directory",
                "List contents of a directory. Returns files and subdirectories "
                "with size and modification time.",
                {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative directory path.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Recursively list subdirectories (default: false).",
                    },
                },
                required=["path"],
            ),
            self._list_directory,
        ))

        # Jupyter notebook operations
        self._add(Tool(
            _spec(
                "create_notebook",
                "Create a new Jupyter notebook with specified cells. Returns the "
                "path to the created .ipynb file.",
                {
                    "path": {
                        "type": "string",
                        "description": "Path for the new notebook (e.g., 'analysis.ipynb').",
                    },
                    "cells": {
                        "type": "array",
                        "description": "Array of cell objects with 'type' (code/markdown) "
                                       "and 'source' (cell content).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["code", "markdown"],
                                },
                                "source": {"type": "string"},
                            },
                            "required": ["type", "source"],
                        },
                    },
                },
                required=["path", "cells"],
            ),
            self._create_notebook,
        ))

        self._add(Tool(
            _spec(
                "run_notebook",
                "Execute a Jupyter notebook and return cell outputs. Uses the "
                "current Python kernel. WARNING: Executes arbitrary code.",
                {
                    "path": {
                        "type": "string",
                        "description": "Path to the notebook to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max seconds per cell (default: 60).",
                    },
                },
                required=["path"],
            ),
            self._run_notebook,
        ))

        self._add(Tool(
            _spec(
                "read_notebook",
                "Read a Jupyter notebook structure and cell outputs without "
                "executing it. Returns cells with their source and any stored outputs.",
                {
                    "path": {
                        "type": "string",
                        "description": "Path to the notebook.",
                    },
                },
                required=["path"],
            ),
            self._read_notebook,
        ))

    # ── handlers ────────────────────────────────────────────────────────

    def _list_tools(self, args: dict) -> dict:
        """List all available tools from all toolsets."""
        verbose = args.get("verbose", False)
        filter_toolset = args.get("toolset")
        tools_info: list[dict] = []
        
        # Add tools from this toolset
        for tool in self._tools.values():
            if filter_toolset and "utility" != filter_toolset:
                continue
            info = {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "toolset": "utility",
            }
            if verbose:
                info["parameters"] = tool.spec.input_schema.get("properties", {})
                info["required"] = tool.spec.input_schema.get("required", [])
            tools_info.append(info)
        
        # Add tools from other registered toolsets
        for ts in self._other_toolsets:
            toolset_name = type(ts).__name__.replace("Tools", "").lower()
            if filter_toolset and toolset_name != filter_toolset:
                continue
            for spec in ts.specs():
                info = {
                    "name": spec.name,
                    "description": spec.description,
                    "toolset": toolset_name,
                }
                if verbose:
                    info["parameters"] = spec.input_schema.get("properties", {})
                    info["required"] = spec.input_schema.get("required", [])
                tools_info.append(info)
        
        # Group by toolset for better readability
        by_toolset = {}
        for t in tools_info:
            ts = t["toolset"]
            if ts not in by_toolset:
                by_toolset[ts] = []
            by_toolset[ts].append(t)
        
        return {
            "count": len(tools_info),
            "by_toolset": by_toolset,
            "toolsets": list(by_toolset.keys()),
        }

    def _read_file(self, args: dict) -> dict:
        """Read a text file from disk."""
        path = Path(args["path"]).expanduser()
        encoding = args.get("encoding", "utf-8")
        
        if not path.exists():
            return {"error": f"File not found: {path}"}
        
        if not path.is_file():
            return {"error": f"Not a file: {path}"}
        
        try:
            content = path.read_text(encoding=encoding)
            return {
                "path": str(path),
                "size": path.stat().st_size,
                "content": content,
                "encoding": encoding,
            }
        except UnicodeDecodeError as e:
            return {"error": f"Encoding error: {e}. Try a different encoding."}

    def _read_image(self, args: dict) -> dict:
        """Read an image file and return as base64 data URI."""
        path = Path(args["path"]).expanduser()
        
        if not path.exists():
            return {"error": f"File not found: {path}"}
        
        if not path.is_file():
            return {"error": f"Not a file: {path}"}
        
        # Determine MIME type from extension
        ext = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        
        mime_type = mime_types.get(ext)
        if not mime_type:
            return {"error": f"Unsupported image format: {ext}"}
        
        try:
            data = path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_uri = f"data:{mime_type};base64,{b64}"
            
            return {
                "path": str(path),
                "size": len(data),
                "mime_type": mime_type,
                "data_uri": data_uri,
                "description": f"Image at {path.name} ({len(data)} bytes)",
            }
        except Exception as e:
            return {"error": f"Failed to read image: {e}"}

    def _write_file(self, args: dict) -> dict:
        """Write text content to a file."""
        path = Path(args["path"]).expanduser()
        content = args["content"]
        encoding = args.get("encoding", "utf-8")
        
        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            path.write_text(content, encoding=encoding)
            
            return {
                "path": str(path),
                "size": path.stat().st_size,
                "encoding": encoding,
                "status": "written",
            }
        except Exception as e:
            return {"error": f"Failed to write file: {e}"}

    def _list_directory(self, args: dict) -> dict:
        """List directory contents."""
        path = Path(args["path"]).expanduser()
        recursive = args.get("recursive", False)
        
        if not path.exists():
            return {"error": f"Directory not found: {path}"}
        
        if not path.is_dir():
            return {"error": f"Not a directory: {path}"}
        
        try:
            entries: list[dict] = []
            
            pattern = "**/*" if recursive else "*"
            for item in sorted(path.glob(pattern)):
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "path": str(item.relative_to(path)),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": stat.st_mtime,
                })
            
            return {
                "path": str(path),
                "count": len(entries),
                "entries": entries,
            }
        except Exception as e:
            return {"error": f"Failed to list directory: {e}"}

    def _create_notebook(self, args: dict) -> dict:
        """Create a new Jupyter notebook."""
        path = Path(args["path"]).expanduser()
        cells_data = args["cells"]
        
        try:
            # Create a new notebook
            nb = nbformat.v4.new_notebook()
            
            # Add EEG session metadata
            if self._session_metadata:
                nb.metadata["pieeg"] = self._session_metadata.copy()
            
            # Add timestamp
            from datetime import datetime
            nb.metadata["created"] = datetime.now().isoformat()
            
            # Add header cell with session info if metadata available
            if self._session_metadata:
                header_lines = ["# PiEEG Analysis Notebook", "", ""]
                if "created" in nb.metadata:
                    header_lines.append(f"**Created:** {nb.metadata['created']}")
                if "stream_name" in self._session_metadata:
                    header_lines.append(f"**Stream:** {self._session_metadata['stream_name']}")
                if "channels" in self._session_metadata:
                    ch = self._session_metadata["channels"]
                    header_lines.append(f"**Channels:** {ch['count']} ({', '.join(ch.get('labels', [])[:4])}{', ...' if ch.get('count', 0) > 4 else ''})")
                if "sample_rate" in self._session_metadata:
                    header_lines.append(f"**Sampling Rate:** {self._session_metadata['sample_rate']:.0f} Hz")
                if "mains_hz" in self._session_metadata:
                    header_lines.append(f"**Mains Frequency:** {self._session_metadata['mains_hz']} Hz")
                header_lines.append("")
                header_lines.append("---")
                header_lines.append("")
                
                header_cell = nbformat.v4.new_markdown_cell("\n".join(header_lines))
                nb.cells.append(header_cell)
            
            # Add user cells
            for cell_data in cells_data:
                cell_type = cell_data["type"]
                source = cell_data["source"]
                
                if cell_type == "code":
                    cell = nbformat.v4.new_code_cell(source)
                elif cell_type == "markdown":
                    cell = nbformat.v4.new_markdown_cell(source)
                else:
                    return {"error": f"Invalid cell type: {cell_type}"}
                
                nb.cells.append(cell)
            
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the notebook
            with open(path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            
            return {
                "path": str(path),
                "cells": len(nb.cells),
                "metadata": dict(nb.metadata),
                "status": "created",
            }
        except Exception as e:
            return {"error": f"Failed to create notebook: {e}"}

    def _run_notebook(self, args: dict) -> dict:
        """Execute a Jupyter notebook."""
        path = Path(args["path"]).expanduser()
        timeout = args.get("timeout", 60)
        
        if not path.exists():
            return {"error": f"Notebook not found: {path}"}
        
        try:
            # Read the notebook
            with open(path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            
            # Determine kernel name - try python3, fallback to python
            import sys
            kernel_name = "python3" if sys.platform != "win32" else "python"
            
            # Execute it
            ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
            try:
                ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
            except Exception as kernel_err:
                # If kernel fails, try with no kernel name (use default)
                ep = ExecutePreprocessor(timeout=timeout)
                ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
            
            # Extract outputs
            outputs: list[dict] = []
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    cell_outputs = []
                    for output in cell.get("outputs", []):
                        if output.output_type == "stream":
                            cell_outputs.append({
                                "type": "stream",
                                "name": output.name,
                                "text": output.text,
                            })
                        elif output.output_type == "execute_result":
                            cell_outputs.append({
                                "type": "result",
                                "data": output.data,
                            })
                        elif output.output_type == "error":
                            cell_outputs.append({
                                "type": "error",
                                "name": output.ename,
                                "value": output.evalue,
                                "traceback": output.traceback,
                            })
                    
                    outputs.append({
                        "cell": i,
                        "source": cell.source[:100] + "..." if len(cell.source) > 100 else cell.source,
                        "outputs": cell_outputs,
                    })
            
            # Save the executed notebook
            with open(path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            
            return {
                "path": str(path),
                "status": "executed",
                "cells": len(nb.cells),
                "outputs": outputs,
            }
        except Exception as e:
            return {"error": f"Failed to execute notebook: {e}"}

    def _read_notebook(self, args: dict) -> dict:
        """Read notebook structure without executing."""
        path = Path(args["path"]).expanduser()
        
        if not path.exists():
            return {"error": f"Notebook not found: {path}"}
        
        try:
            with open(path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            
            cells: list[dict] = []
            for i, cell in enumerate(nb.cells):
                cell_data = {
                    "index": i,
                    "type": cell.cell_type,
                    "source": cell.source,
                }
                
                if cell.cell_type == "code":
                    cell_data["outputs"] = [
                        {
                            "type": out.output_type,
                            "content": str(out.get("text", out.get("data", {}))),
                        }
                        for out in cell.get("outputs", [])
                    ]
                
                cells.append(cell_data)
            
            return {
                "path": str(path),
                "cells": cells,
                "metadata": nb.metadata,
            }
        except Exception as e:
            return {"error": f"Failed to read notebook: {e}"}
