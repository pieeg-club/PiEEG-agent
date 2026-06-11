import { useState, useEffect } from "react";
import {
  loadConversations,
  deleteConversation,
  type Conversation,
} from "../util/conversations";
import { toast } from "./Toast";

interface ConversationListProps {
  currentId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onClose: () => void;
}

export function ConversationList({ currentId, onSelect, onNew, onClose }: ConversationListProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);

  const refresh = () => {
    setConversations(loadConversations());
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleDelete = (id: string, title: string) => {
    if (!confirm(`Delete "${title}"?`)) return;
    deleteConversation(id);
    toast.success("Conversation deleted");
    refresh();
    if (currentId === id) {
      onNew();
    }
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } else if (days === 1) {
      return "Yesterday";
    } else if (days < 7) {
      return `${days} days ago`;
    } else {
      return date.toLocaleDateString([], { month: "short", day: "numeric" });
    }
  };

  return (
    <>
      <div className="sidebar-overlay" onClick={onClose} />
      <div className="conversation-sidebar">
        <div className="conversation-header">
          <h2>Conversations</h2>
          <button className="close-btn" onClick={onClose} title="Close">
            ✕
          </button>
        </div>
      
      <button className="new-conversation-btn" onClick={onNew}>
        + New Conversation
      </button>
      
      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="empty-state">No saved conversations</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${conv.id === currentId ? "active" : ""}`}
            >
              <button
                className="conversation-select"
                onClick={() => {
                  onSelect(conv.id);
                  onClose();
                }}
              >
                <div className="conversation-title">{conv.title}</div>
                <div className="conversation-date">{formatDate(conv.timestamp)}</div>
              </button>
              <button
                className="conversation-delete"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(conv.id, conv.title);
                }}
                title="Delete"
              >
                🗑
              </button>
            </div>
          ))
        )}
      </div>
    </div>
    </>
  );
}
