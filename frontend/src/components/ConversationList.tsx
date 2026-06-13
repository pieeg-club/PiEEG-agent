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
  isOpen: boolean;
  onToggle: () => void;
}

export function ConversationList({ currentId, onSelect, onNew, isOpen, onToggle }: ConversationListProps) {
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

  if (!isOpen) return null;

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
    <div className="conversation-sidebar">
      <div className="conversation-header">
        <h2>Conversations</h2>
        <button className="close-btn" onClick={onToggle} title="Close sidebar">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>
      
      <button className="new-conversation-btn" onClick={onNew}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19"></line>
          <line x1="5" y1="12" x2="19" y2="12"></line>
        </svg>
        <span>New Conversation</span>
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
                onClick={() => onSelect(conv.id)}
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
                title="Delete conversation"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  <line x1="10" y1="11" x2="10" y2="17"></line>
                  <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
