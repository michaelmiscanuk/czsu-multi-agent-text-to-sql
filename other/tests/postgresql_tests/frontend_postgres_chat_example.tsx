/**
 * Example React component demonstrating how to replace IndexedDB usage 
 * with PostgreSQL-based chat management via API calls.
 * 
 * This replaces the IndexedDB functions in frontend/src/components/utils.ts
 */

"use client";
import React, { useState, useEffect } from 'react';
import { useSession } from "next-auth/react";

// Types for PostgreSQL-based chat management
interface ChatThreadMeta {
  thread_id: string;
  latest_timestamp: string; // ISO string from PostgreSQL
  run_count: number;
}

interface ChatMessage {
  id: string;
  threadId: string;
  user: string;
  content: string;
  isUser: boolean;
  createdAt: number;
  error?: string;
  meta?: Record<string, any>;
  queriesAndResults?: [string, string][];
  isLoading?: boolean;
  startedAt?: number;
  isError?: boolean;
}

// API functions to replace IndexedDB operations
class PostgresChatAPI {
  private apiBase: string;
  private getAuthHeaders: () => Promise<HeadersInit>;

  constructor(apiBase: string, getAuthHeaders: () => Promise<HeadersInit>) {
    this.apiBase = apiBase;
    this.getAuthHeaders = getAuthHeaders;
  }

  /**
   * Get all chat threads for the current user from PostgreSQL
   * Replaces: listThreads(user) from IndexedDB utils
   */
  async listThreads(): Promise<ChatThreadMeta[]> {
    try {
      const headers = await this.getAuthHeaders();
      const response = await fetch(`${this.apiBase}/chat-threads`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        throw new Error(`Failed to get chat threads: ${response.status}`);
      }

      const threads = await response.json();
      console.log('[PostgresChatAPI] Retrieved threads:', threads);
      return threads;
    } catch (error) {
      console.error('[PostgresChatAPI] Error getting threads:', error);
      return [];
    }
  }

  /**
   * Delete a chat thread and all its data
   * Replaces: deleteThread(user, id) from IndexedDB utils
   */
  async deleteThread(threadId: string): Promise<boolean> {
    try {
      const headers = await this.getAuthHeaders();
      const response = await fetch(`${this.apiBase}/chat/${threadId}`, {
        method: 'DELETE',
        headers
      });

      if (!response.ok) {
        console.error(`Failed to delete thread: ${response.status}`);
        return false;
      }

      const result = await response.json();
      console.log('[PostgresChatAPI] Thread deleted:', result);
      return true;
    } catch (error) {
      console.error('[PostgresChatAPI] Error deleting thread:', error);
      return false;
    }
  }

  /**
   * Send a message and create thread run entry
   * This integrates with the existing /analyze endpoint which now automatically
   * creates thread run entries in PostgreSQL
   */
  async sendMessage(threadId: string, prompt: string): Promise<any> {
    try {
      const headers = await this.getAuthHeaders();
      const response = await fetch(`${this.apiBase}/analyze`, {
        method: 'POST',
        headers: {
          ...headers,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt, thread_id: threadId })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }

      const result = await response.json();
      console.log('[PostgresChatAPI] Message sent, run_id:', result.run_id);
      return result;
    } catch (error) {
      console.error('[PostgresChatAPI] Error sending message:', error);
      throw error;
    }
  }
}

// Example React component using the new PostgreSQL-based system
export default function PostgresChatExample() {
  const { data: session } = useSession();
  const [threads, setThreads] = useState<ChatThreadMeta[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

  // Initialize API helper
  const chatAPI = new PostgresChatAPI(API_BASE, async () => {
    const freshSession = await getSession();
    return {
      'Authorization': `Bearer ${freshSession?.id_token}`
    };
  });

  // Load chat threads from PostgreSQL instead of IndexedDB
  useEffect(() => {
    if (!session?.user?.email) return;
    
    loadThreadsFromPostgreSQL();
  }, [session?.user?.email]);

  const loadThreadsFromPostgreSQL = async () => {
    try {
      const postgresThreads = await chatAPI.listThreads();
      setThreads(postgresThreads);
      
      // If no active thread and we have threads, select the most recent one
      if (!activeThreadId && postgresThreads.length > 0) {
        setActiveThreadId(postgresThreads[0].thread_id);
      }
    } catch (error) {
      console.error('Failed to load threads from PostgreSQL:', error);
    }
  };

  // Handle new chat creation
  const handleNewChat = () => {
    // Generate new thread ID
    const newThreadId = `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setActiveThreadId(newThreadId);
    setMessages([]); // Clear messages for new chat
    
    // Note: Thread will be created automatically when first message is sent
    // via the /analyze endpoint which now creates thread run entries
  };

  // Handle chat deletion using PostgreSQL
  const handleDeleteChat = async (threadId: string) => {
    try {
      const success = await chatAPI.deleteThread(threadId);
      
      if (success) {
        // Reload threads from PostgreSQL
        await loadThreadsFromPostgreSQL();
        
        // If we deleted the active thread, switch to another one
        if (activeThreadId === threadId) {
          const remainingThreads = threads.filter(t => t.thread_id !== threadId);
          setActiveThreadId(remainingThreads.length > 0 ? remainingThreads[0].thread_id : null);
        }
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
    }
  };

  // Handle message sending with PostgreSQL integration
  const handleSendMessage = async () => {
    if (!currentMessage.trim() || !activeThreadId) return;
    
    setIsLoading(true);
    
    try {
      // Send message via API (this automatically creates thread run entry)
      const result = await chatAPI.sendMessage(activeThreadId, currentMessage);
      
      // Reload threads to get updated timestamp and run count
      await loadThreadsFromPostgreSQL();
      
      // Handle the response (update messages state, etc.)
      console.log('Message sent successfully, run_id:', result.run_id);
      
      // Clear the input
      setCurrentMessage("");
      
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar with threads from PostgreSQL */}
      <div className="w-1/4 bg-gray-100 p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold">Chat Threads</h2>
          <button 
            onClick={handleNewChat}
            className="px-3 py-1 bg-blue-500 text-white rounded text-sm"
          >
            New Chat
          </button>
        </div>
        
        <div className="space-y-2">
          {threads.map(thread => (
            <div 
              key={thread.thread_id}
              className={`p-3 rounded cursor-pointer ${
                activeThreadId === thread.thread_id ? 'bg-blue-200' : 'bg-white'
              }`}
              onClick={() => setActiveThreadId(thread.thread_id)}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">
                    Thread {thread.thread_id.slice(-8)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(thread.latest_timestamp).toLocaleDateString()}
                  </div>
                  <div className="text-xs text-gray-400">
                    {thread.run_count} messages
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteChat(thread.thread_id);
                  }}
                  className="ml-2 text-red-500 hover:text-red-700 text-sm"
                >
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>
        
        {threads.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            No chats yet. Create your first chat!
          </div>
        )}
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        {activeThreadId ? (
          <>
            <div className="flex-1 p-4 overflow-y-auto">
              {/* Messages would go here */}
              <div className="text-center text-gray-500">
                Chat content for thread: {activeThreadId}
              </div>
            </div>
            
            {/* Message input */}
            <div className="p-4 border-t">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Type your message..."
                  className="flex-1 p-2 border rounded"
                  disabled={isLoading}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !currentMessage.trim()}
                  className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
                >
                  Send
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            Select a chat or create a new one to get started
          </div>
        )}
      </div>
    </div>
  );
}

// Helper function to get session (you'll need to import this)
async function getSession() {
  // This should be imported from next-auth/react
  // return await getSession();
  return null; // placeholder
}

/**
 * Migration Guide: Replacing IndexedDB with PostgreSQL
 * 
 * OLD IndexedDB approach:
 * - listThreads(user) -> stored locally in browser
 * - saveThread(meta) -> stored locally in browser  
 * - deleteThread(user, id) -> deleted locally only
 * - Data lost on browser clear/different device
 * 
 * NEW PostgreSQL approach:
 * - GET /chat-threads -> fetches from PostgreSQL
 * - Thread creation happens automatically on first message
 * - DELETE /chat/{thread_id} -> deletes from PostgreSQL
 * - Data persists across devices and browser clears
 * 
 * Key benefits:
 * 1. Data persistence across devices
 * 2. No more browser storage limitations
 * 3. Server restart doesn't lose state
 * 4. Better performance and reliability
 * 5. Proper user isolation
 * 
 * Migration steps:
 * 1. Replace all IndexedDB function calls with API calls
 * 2. Update thread loading to use GET /chat-threads
 * 3. Update thread deletion to use DELETE /chat/{thread_id}
 * 4. Remove IndexedDB utility functions
 * 5. Test thoroughly with multiple users and devices
 */ 