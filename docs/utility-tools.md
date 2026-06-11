# Utility Tools — Extended Capabilities

The PiEEG-agent now includes general-purpose utility tools for file I/O, Jupyter notebook automation, and tool discovery.

## Quick Reference

### Tool Discovery

**`list_tools`** — Discover all available tools across all toolsets

```python
# Example query:
"What tools do you have available?"
"Show me all capabilities"
```

Returns:
- Tool names and descriptions
- Parameter schemas and requirements
- Toolset categorization (neural, decode, documentation, actuator, utility)

### File System Operations

**`read_file`** — Read text files (code, logs, configs, CSV, JSON, etc.)

```python
# Example queries:
"Read the config.json file"
"Show me the contents of recording_log.csv"
"What's in the latest EEG recording metadata?"
```

Parameters:
- `path`: Absolute or relative file path
- `encoding`: Optional text encoding (default: utf-8)

**`write_file`** — Write text content to a file

```python
# Example queries:
"Save this analysis to results.txt"
"Write a summary of the session to summary.md"
"Create a config.json with these settings"
```

Parameters:
- `path`: File path (creates parent directories if needed)
- `content`: Text content to write
- `encoding`: Optional encoding (default: utf-8)

**`list_directory`** — List directory contents with metadata

```python
# Example queries:
"What files are in the recordings folder?"
"List all CSV files in the current directory"
"Show me the contents of /data recursively"
```

Parameters:
- `path`: Directory path
- `recursive`: Optional boolean for recursive listing (default: false)

**`read_image`** — Read and display image files

```python
# Example queries:
"Show me the spectrogram.png"
"Display the latest brain activity visualization"
"View the electrode placement diagram"
```

Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP

Returns base64 data URI for viewing in compatible interfaces.

### Jupyter Notebook Automation

**`create_notebook`** — Create a new Jupyter notebook programmatically

```python
# Example query:
"Create a notebook that analyzes the EEG data from session X"
```

Parameters:
- `path`: Path for the new .ipynb file
- `cells`: Array of cell objects with:
  - `type`: "code" or "markdown"
  - `source`: Cell content (code or markdown text)

Example usage in conversation:
```
User: "Create a notebook to analyze my focus patterns"

Agent: (creates notebook with cells for):
1. Markdown: "# Focus Pattern Analysis"
2. Code: import libraries
3. Code: load session data
4. Code: compute statistics
5. Code: generate visualizations
```

**`run_notebook`** — Execute a Jupyter notebook and capture outputs

```python
# Example queries:
"Run the analysis.ipynb notebook"
"Execute my spectral analysis notebook and show results"
```

Parameters:
- `path`: Path to the notebook
- `timeout`: Optional timeout in seconds per cell (default: 60)

Returns:
- Execution status
- Cell outputs (stdout, results, errors)
- Saved back to the original file with outputs

**`read_notebook`** — Read notebook structure without executing

```python
# Example queries:
"What's in the analysis notebook?"
"Show me the structure of my EEG processing pipeline"
```

Returns:
- Cell types and source code
- Any previously stored outputs
- Notebook metadata

## Usage Examples

### Example 1: Automated EEG Analysis Workflow

```
User: "Create a notebook that analyzes my last recording session, computes band powers, and saves a summary"

Agent: 
1. Uses list_sessions to find the latest session
2. Creates a notebook with:
   - Data loading cells
   - Band power computation
   - Statistical summary
   - File output
3. Runs the notebook
4. Returns the results and summary file path
```

### Example 2: Pattern Comparison Report

```
User: "Compare my 'focus' and 'relax' patterns and save a detailed report"

Agent:
1. Uses compare_sessions to get Cohen's d statistics
2. Uses write_file to create a markdown report with:
   - Effect sizes for each feature
   - Interpretation of results
   - Recommendations
3. Optionally creates a visualization notebook
```

### Example 3: Tool Discovery

```
User: "What can you do with files and data?"

Agent: Uses list_tools and filters for utility/file-related tools, then explains:
- File reading/writing capabilities
- Image viewing
- Directory exploration
- Notebook automation
```

## Integration with Existing Tools

The utility tools work seamlessly with neural sensing tools:

1. **Session → Notebook → File**
   - Record a session with `record_session`
   - Analyze it in a notebook with `create_notebook` + `run_notebook`
   - Save results with `write_file`

2. **Pattern → Analysis → Report**
   - Train patterns with `start_pattern_training` + `record_segment`
   - Create comparative analysis notebook
   - Generate report with visualizations

3. **Documentation → Action**
   - Search PiEEG docs with `search_docs`
   - Save relevant snippets with `write_file`
   - Create tutorial notebooks for future reference

## Safety & Best Practices

### File Operations
- Paths are expanded with `~` support
- Parent directories are created automatically on write
- Encoding defaults to UTF-8 but can be overridden
- Errors return descriptive messages rather than crashing

### Notebook Execution
- **WARNING**: `run_notebook` executes arbitrary code
- Only run notebooks from trusted sources
- Timeout prevents runaway cells (default 60s/cell)
- Execution happens in the current Python environment
- All outputs are captured and returned

### Tool Discovery
- `list_tools` reflects the **current session's capabilities**
- Actuator tools only appear if `--allow-actions` is enabled
- Use this to verify what's available before asking the agent to act

## Technical Details

### Architecture
- UtilityTools implements the same `Toolset` protocol as neural/actuator tools
- Registered with `CombinedToolset` alongside other tool groups
- Tool discovery is self-referential (UtilityTools sees all other toolsets)

### Dependencies
- `nbformat>=5.9` — Jupyter notebook format library
- `nbconvert>=7.0` — Notebook execution engine

### Error Handling
- All tools return `{"error": "..."}` dicts on failure
- No exceptions escape to the copilot loop
- Unknown tools return available tool list
- File not found, permission errors, encoding issues all handled gracefully

### Future Extensions

Potential additions:
- `execute_python` — Run arbitrary Python snippets (with sandboxing)
- `plot_spectrum` — Generate and save spectrograms programmatically
- `export_session_data` — Dump session recordings to CSV/Parquet
- `search_files` — Grep/semantic search across local files
- `watch_directory` — Monitor for new recordings and auto-process

---

**Remember**: These tools extend the agent's capabilities beyond neural sensing. Use `list_tools` anytime to see what's available in your current session!
