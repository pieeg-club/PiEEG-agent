// Conversation persistence using localStorage.
// Each conversation is stored with metadata (id, title, timestamp) and full message history.

import type { ChatMessage } from "../hooks/useChatSocket";

export interface Conversation {
  id: string;
  title: string;
  timestamp: number; // ms since epoch
  messages: ChatMessage[];
}

const STORAGE_KEY = "pieeg-conversations";
const CURRENT_ID_KEY = "pieeg-current-conversation-id";

// Generate a title from the first user message (up to 50 chars)
function generateTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser || firstUser.parts.length === 0) {
    return "New conversation";
  }
  const text = firstUser.parts
    .filter((p) => p.kind === "text")
    .map((p) => (p as { text: string }).text)
    .join(" ")
    .trim();
  return text.length > 50 ? text.slice(0, 47) + "..." : text || "New conversation";
}

// Load all conversations from localStorage
export function loadConversations(): Conversation[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
}

// Save all conversations to localStorage
function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch (err) {
    console.error("Failed to save conversations:", err);
  }
}

// Save or update a conversation
export function saveConversation(id: string, messages: ChatMessage[]): void {
  if (messages.length === 0) return; // Don't save empty conversations
  
  const conversations = loadConversations();
  const index = conversations.findIndex((c) => c.id === id);
  const timestamp = Date.now();
  const title = generateTitle(messages);
  
  if (index >= 0) {
    // Update existing
    conversations[index] = { id, title, timestamp, messages };
  } else {
    // Create new
    conversations.push({ id, title, timestamp, messages });
  }
  
  // Sort by timestamp descending (newest first)
  conversations.sort((a, b) => b.timestamp - a.timestamp);
  
  saveConversations(conversations);
}

// Load a specific conversation by ID
export function loadConversation(id: string): Conversation | null {
  const conversations = loadConversations();
  return conversations.find((c) => c.id === id) || null;
}

// Delete a conversation
export function deleteConversation(id: string): void {
  const conversations = loadConversations();
  const filtered = conversations.filter((c) => c.id !== id);
  saveConversations(filtered);
  
  // Clear current ID if it was deleted
  if (getCurrentConversationId() === id) {
    clearCurrentConversationId();
  }
}

// Generate a new unique conversation ID
export function generateConversationId(): string {
  return `conv_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// Get/set the current active conversation ID
export function getCurrentConversationId(): string | null {
  return sessionStorage.getItem(CURRENT_ID_KEY);
}

export function setCurrentConversationId(id: string): void {
  sessionStorage.setItem(CURRENT_ID_KEY, id);
}

export function clearCurrentConversationId(): void {
  sessionStorage.removeItem(CURRENT_ID_KEY);
}
