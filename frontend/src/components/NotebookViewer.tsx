import { useEffect, useState } from "react";
import { marked } from "marked";
import { api } from "../api";
import type { NotebookData } from "../types";

interface NotebookViewerProps {
  notebookPath: string;
  onClose: () => void;
}

export function NotebookViewer({ notebookPath, onClose }: NotebookViewerProps) {
  const [notebook, setNotebook] = useState<NotebookData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch notebook data via the read_notebook tool through chat context
    // For now, we'll use the API directly
    setLoading(true);
    setError(null);
    
    // Mock fetch - in reality this would call the backend
    // For the complete implementation, we'd need an endpoint or use the tool system
    fetch(`/api/notebook?path=${encodeURIComponent(notebookPath)}`)
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
        } else {
          setNotebook(data);
        }
      })
      .catch(err => {
        setError("Failed to load notebook");
      })
      .finally(() => setLoading(false));
  }, [notebookPath]);

  if (loading) {
    return (
      <div className="notebook-modal">
        <div className="notebook-container">
          <div className="notebook-header">
            <h2>Loading notebook...</h2>
            <button className="close-btn" onClick={onClose}>✕</button>
          </div>
          <div className="notebook-loading">
            <div className="spinner"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !notebook) {
    return (
      <div className="notebook-modal">
        <div className="notebook-container">
          <div className="notebook-header">
            <h2>Error</h2>
            <button className="close-btn" onClick={onClose}>✕</button>
          </div>
          <div className="notebook-error">
            <p>{error || "Failed to load notebook"}</p>
          </div>
        </div>
      </div>
    );
  }

  const notebookName = notebookPath.split(/[/\\]/).pop() || "notebook.ipynb";

  return (
    <div className="notebook-modal" onClick={onClose}>
      <div className="notebook-container" onClick={(e) => e.stopPropagation()}>
        <div className="notebook-header">
          <div className="notebook-title">
            <span className="notebook-icon">📓</span>
            <h2>{notebookName}</h2>
          </div>
          <div className="notebook-actions">
            <button
              className="notebook-btn"
              onClick={() => {
                const blob = new Blob([JSON.stringify(notebook, null, 2)], {
                  type: "application/json",
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = notebookName;
                a.click();
                URL.revokeObjectURL(url);
              }}
              title="Download notebook"
            >
              ⬇ Download
            </button>
            <button className="close-btn" onClick={onClose}>✕</button>
          </div>
        </div>

        {notebook.metadata && notebook.metadata.pieeg && (
          <div className="notebook-meta">
            {notebook.metadata.created && (
              <span className="meta-item">
                <strong>Created:</strong> {new Date(notebook.metadata.created).toLocaleString()}
              </span>
            )}
            {notebook.metadata.pieeg.stream_name && (
              <span className="meta-item">
                <strong>Stream:</strong> {notebook.metadata.pieeg.stream_name}
              </span>
            )}
            {notebook.metadata.pieeg.channels && (
              <span className="meta-item">
                <strong>Channels:</strong> {notebook.metadata.pieeg.channels.count}
              </span>
            )}
            {notebook.metadata.pieeg.sample_rate && (
              <span className="meta-item">
                <strong>Rate:</strong> {notebook.metadata.pieeg.sample_rate} Hz
              </span>
            )}
          </div>
        )}

        <div className="notebook-cells">
          {notebook.cells.map((cell, idx) => (
            <NotebookCell key={idx} cell={cell} index={idx} />
          ))}
        </div>
      </div>
    </div>
  );
}

interface NotebookCellProps {
  cell: NotebookData["cells"][0];
  index: number;
}

function NotebookCell({ cell, index }: NotebookCellProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (cell.type === "markdown") {
    const html = marked.parse(cell.source, { async: false }) as string;
    return (
      <div className="notebook-cell markdown-cell">
        <div
          className="cell-content markdown-content"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
    );
  }

  if (cell.type === "code") {
    const hasOutput = cell.outputs && cell.outputs.length > 0;

    return (
      <div className="notebook-cell code-cell">
        <div className="cell-header">
          <span className="cell-label">In [{index + 1}]</span>
          {hasOutput && (
            <button
              className="cell-collapse-btn"
              onClick={() => setCollapsed(!collapsed)}
              title={collapsed ? "Expand" : "Collapse"}
            >
              {collapsed ? "▸" : "▾"}
            </button>
          )}
        </div>
        <pre className="cell-code">
          <code>{cell.source}</code>
        </pre>
        {hasOutput && !collapsed && (
          <div className="cell-outputs">
            {cell.outputs?.map((output, outIdx) => (
              <CellOutput key={outIdx} output={output} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return null;
}

interface CellOutputProps {
  output: NonNullable<NotebookData["cells"][0]["outputs"]>[0];
}

function CellOutput({ output }: CellOutputProps) {
  if (output.type === "stream") {
    return (
      <div className="cell-output stream-output">
        <pre>{typeof output.content === "string" ? output.content : JSON.stringify(output.content, null, 2)}</pre>
      </div>
    );
  }

  if (output.type === "result") {
    const content = output.content;
    
    // Handle different result types
    if (typeof content === "object" && content !== null) {
      // Check for image data
      const data = content as any;
      if (data["image/png"]) {
        return (
          <div className="cell-output image-output">
            <img src={`data:image/png;base64,${data["image/png"]}`} alt="Plot" />
          </div>
        );
      }
      if (data["image/jpeg"]) {
        return (
          <div className="cell-output image-output">
            <img src={`data:image/jpeg;base64,${data["image/jpeg"]}`} alt="Plot" />
          </div>
        );
      }
      if (data["text/html"]) {
        return (
          <div
            className="cell-output html-output"
            dangerouslySetInnerHTML={{ __html: data["text/html"] }}
          />
        );
      }
      if (data["text/plain"]) {
        return (
          <div className="cell-output text-output">
            <pre>{data["text/plain"]}</pre>
          </div>
        );
      }
      // Fallback: show JSON
      return (
        <div className="cell-output json-output">
          <pre>{JSON.stringify(content, null, 2)}</pre>
        </div>
      );
    }

    return (
      <div className="cell-output text-output">
        <pre>{String(content)}</pre>
      </div>
    );
  }

  if (output.type === "error") {
    const error = output.content as any;
    return (
      <div className="cell-output error-output">
        <div className="error-name">{error.name || "Error"}</div>
        <div className="error-value">{error.value || String(error)}</div>
        {error.traceback && (
          <pre className="error-traceback">
            {Array.isArray(error.traceback) ? error.traceback.join("\n") : error.traceback}
          </pre>
        )}
      </div>
    );
  }

  return (
    <div className="cell-output unknown-output">
      <pre>{JSON.stringify(output, null, 2)}</pre>
    </div>
  );
}
