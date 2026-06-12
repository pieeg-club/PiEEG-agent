"""Lightweight RAG layer for session data retrieval.

Instead of dumping full session JSON into the conversation history, this module
enables semantic retrieval of session information. Session summaries are stored
with metadata that allows targeted queries.

The workflow:
  1. When a session is recorded, store a compressed summary + metadata
  2. When analyze_session is called, return a compact summary
  3. If the LLM needs specific details (e.g., "which channels had highest alpha?"),
     it can query the session store with targeted questions

This reduces context window pressure while preserving the ability to retrieve
detailed information on demand.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("pieeg.agent.rag")


@dataclass
class SessionIndex:
    """Indexed metadata for semantic retrieval."""
    
    label: str
    timestamp: float
    duration_s: float
    dominant_band: str
    
    # High-level metrics for filtering
    mean_quality: float
    total_artifacts: int
    
    # Searchable text description
    description: str
    
    # Reference to full data
    data_path: Path | None = None


class SessionRAG:
    """Lightweight retrieval-augmented generation for session data.
    
    This is a simple implementation that doesn't require external vector databases.
    It uses keyword matching and metadata filtering to retrieve relevant session
    information.
    """
    
    def __init__(self):
        self._index: dict[str, SessionIndex] = {}
    
    def index_session(self, label: str, session_data: dict, data_path: Path | None = None):
        """Add a session to the searchable index.
        
        Args:
            label: Session label
            session_data: Full session data dict
            data_path: Optional path to the full data file
        """
        # Extract metadata
        indices = session_data.get("indices", {})
        quality = session_data.get("signal_quality", {})
        artifacts = session_data.get("artifacts", {})
        
        # Build searchable description
        description = self._build_description(session_data)
        
        index_entry = SessionIndex(
            label=label,
            timestamp=session_data.get("timestamp", 0),
            duration_s=session_data.get("duration_s", 0),
            dominant_band=session_data.get("dominant_band", ""),
            mean_quality=quality.get("mean", 0),
            total_artifacts=sum(artifacts.values()) if isinstance(artifacts, dict) else 0,
            description=description,
            data_path=data_path,
        )
        
        self._index[label] = index_entry
        logger.debug(f"Indexed session '{label}' with {len(description)} chars description")
    
    def _build_description(self, session_data: dict) -> str:
        """Build a searchable text description of the session."""
        parts = []
        
        # Add dominant band
        if "dominant_band" in session_data:
            parts.append(f"dominant: {session_data['dominant_band']}")
        
        # Add indices
        indices = session_data.get("indices", {})
        if indices:
            for name, value in indices.items():
                level = "high" if value > 0.7 else "medium" if value > 0.4 else "low"
                parts.append(f"{name}: {level}")
        
        # Add band power summary
        if "band_powers" in session_data:
            bp = session_data["band_powers"]
            if isinstance(bp, dict):
                bands = []
                for band, values in bp.items():
                    if isinstance(values, dict) and "mean" in values:
                        bands.append(f"{band}={values['mean']:.2f}")
                if bands:
                    parts.append(f"bands: {', '.join(bands)}")
        
        # Add artifact summary
        artifacts = session_data.get("artifacts", {})
        if artifacts:
            art_parts = [f"{k}={v}" for k, v in artifacts.items() if v > 0]
            if art_parts:
                parts.append(f"artifacts: {', '.join(art_parts)}")
        
        # Add quality
        quality = session_data.get("signal_quality", {})
        if "mean" in quality:
            q = quality["mean"]
            q_level = "good" if q > 0.7 else "fair" if q > 0.4 else "poor"
            parts.append(f"quality: {q_level}")
        
        return " | ".join(parts)
    
    def get_compressed_summary(self, label: str, session_data: dict) -> dict:
        """Get a compressed summary suitable for LLM context.
        
        Returns a compact representation that preserves semantic meaning
        but reduces token count significantly.
        """
        if "error" in session_data:
            return session_data
        
        # Index the session if not already indexed
        if label not in self._index:
            self.index_session(label, session_data)
        
        index_entry = self._index[label]
        
        # Return compact summary
        summary = {
            "label": label,
            "duration_s": index_entry.duration_s,
            "dominant_band": index_entry.dominant_band,
            "description": index_entry.description,
        }
        
        # Include indices if present
        if "indices" in session_data:
            summary["indices"] = session_data["indices"]
        
        # Include quality summary
        summary["mean_quality"] = index_entry.mean_quality
        summary["total_artifacts"] = index_entry.total_artifacts
        
        # Add retrieval hint
        summary["_retrieval_hint"] = (
            "This is a compressed summary. Full spectral details, per-channel "
            "quality, and connectivity available on request."
        )
        
        return summary
    
    def search(self, query: str, limit: int = 5) -> list[str]:
        """Search for sessions matching a query.
        
        Args:
            query: Natural language query (e.g., "high focus sessions")
            limit: Max number of results
            
        Returns:
            List of session labels ranked by relevance
        """
        # Simple keyword matching for now
        # A more sophisticated version could use embeddings
        query_lower = query.lower()
        keywords = query_lower.split()
        
        # Score each session
        scores = []
        for label, index_entry in self._index.items():
            score = 0
            desc_lower = index_entry.description.lower()
            
            # Count keyword matches
            for keyword in keywords:
                if keyword in label.lower():
                    score += 3  # Label match is strong signal
                if keyword in desc_lower:
                    score += 1
            
            # Bonus for recent sessions (decay over time)
            # (This could be made more sophisticated)
            
            if score > 0:
                scores.append((label, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [label for label, _ in scores[:limit]]
    
    def list_all(self) -> list[dict]:
        """List all indexed sessions with metadata."""
        return [
            {
                "label": entry.label,
                "duration_s": entry.duration_s,
                "dominant_band": entry.dominant_band,
                "mean_quality": entry.mean_quality,
                "description": entry.description,
            }
            for entry in self._index.values()
        ]
    
    def remove(self, label: str) -> bool:
        """Remove a session from the index."""
        if label in self._index:
            del self._index[label]
            return True
        return False
    
    def clear(self):
        """Clear all indexed sessions."""
        self._index.clear()
