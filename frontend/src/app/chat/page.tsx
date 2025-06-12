"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession, getSession, signOut } from "next-auth/react";
import { useChatCache } from '@/contexts/ChatCacheContext';

// Types for PostgreSQL-based chat management
interface ChatThreadMeta {
  thread_id: string;
  latest_timestamp: string;
  run_count: number;
  title: string;
  full_prompt: string; // For tooltip display
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

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  selectionCode?: string | null;
  queriesAndResults?: [string, string][];
  meta?: {
    datasetUrl?: string;
    datasetsUsed?: string[];  // Array of dataset codes actually used in queries
    sqlQuery?: string;
  };
}

const INITIAL_MESSAGE = [
  {
    id: 1,
    content: 'Hi there, how can I help you?',
    isUser: false,
    type: 'message'
  }
];

export default function ChatPage() {
  const { data: session, status, update } = useSession();
  const userEmail = session?.user?.email || null;
  
  // Use the new ChatCache context
  const {
    threads,
    messages,
    activeThreadId,
    isLoading: cacheLoading,
    setThreads,
    setMessages,
    setActiveThreadId,
    addMessage,
    updateMessage,
    addThread,
    removeThread,
    updateThread,
    invalidateCache,
    refreshFromAPI,
    isDataStale,
    setLoading,
    isPageRefresh,
    forceAPIRefresh
  } = useChatCache();
  
  console.log('[ChatPage-DEBUG] 🔄 Component render - Status:', status, 'UserEmail:', !!userEmail, 'IsPageRefresh:', isPageRefresh, 'Timestamp:', new Date().toISOString());
  
  // Show loading while session is being fetched
  if (status === "loading") {
    console.log('[ChatPage-DEBUG] ⏳ Session loading state');
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <div className="text-gray-600">Loading your session...</div>
        </div>
      </div>
    );
  }
  
  // Redirect to login if not authenticated
  if (status === "unauthenticated" || !userEmail) {
    console.log('[ChatPage-DEBUG] ❌ Not authenticated - Status:', status, 'UserEmail:', !!userEmail);
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-gray-600 mb-4">Please sign in to access your chats</div>
          <button 
            onClick={() => window.location.href = '/api/auth/signin'}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Sign In
          </button>
        </div>
      </div>
    );
  }
  
  // Local component state
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [openSQLModalForMsgId, setOpenSQLModalForMsgId] = useState<string | null>(null);
  const [iteration, setIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(2); // default fallback
  const [threadsLoaded, setThreadsLoaded] = useState(false);
  const [threadsLoading, setThreadsLoading] = useState(false);
  
  // Debug logging for state changes
  React.useEffect(() => {
    console.log('[ChatPage-DEBUG] 📊 State Update - threads:', threads.length, 'activeThreadId:', activeThreadId, 'messages:', messages.length, 'threadsLoaded:', threadsLoaded, 'threadsLoading:', threadsLoading);
  }, [threads.length, activeThreadId, messages.length, threadsLoaded, threadsLoading]);
  
  // Track previous chatId and message count for scroll logic
  const prevChatIdRef = React.useRef<string | null>(null);
  const prevMsgCountRef = React.useRef<number>(1);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const sidebarRef = React.useRef<HTMLDivElement>(null);
  
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  // PostgreSQL API functions with new cache context
  const loadThreadsFromPostgreSQL = async () => {
    if (!userEmail) {
      console.log('[ChatPage-loadThreads] ❌ No user email available');
      return;
    }

    console.log('[ChatPage-loadThreads] 🔄 Loading threads from PostgreSQL for user:', userEmail);
    console.log('[ChatPage-loadThreads] 💾 Cache state - Stale:', isDataStale(), 'Threads:', threads.length, 'IsPageRefresh:', isPageRefresh);

    // FORCE API CALL only on actual F5 page refresh to sync localStorage with PostgreSQL
    if (isPageRefresh) {
      console.log('[ChatPage-loadThreads] 🔄 Actual page refresh (F5) detected - forcing fresh API call to sync with PostgreSQL');
      await forceAPIRefresh(); // Clear cache first
    }
    // Use cache if fresh and has data (navigation between menus)
    else if (!isDataStale() && threads.length > 0) {
      console.log('[ChatPage-loadThreads] ✅ Using fresh cache data (navigation between menus)');
      setThreadsLoaded(true);
      return;
    }

    setThreadsLoading(true);
    setLoading(true);

    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      console.log('[ChatPage-loadThreads] 📡 Making API call to PostgreSQL...');
      const response = await fetch(`${API_BASE}/chat-threads`, {
        headers: {
          'Authorization': `Bearer ${freshSession.id_token}`
        }
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('[ChatPage-loadThreads] ✅ Loaded threads from PostgreSQL API:', data.length);
      
      // Update cache through context - this will sync localStorage with PostgreSQL data
      setThreads(data);
      setThreadsLoaded(true);
      
      if (isPageRefresh) {
        console.log('[ChatPage-loadThreads] ✅ F5 refresh completed - localStorage now synced with PostgreSQL');
      } else {
        console.log('[ChatPage-loadThreads] ✅ Navigation completed - cache populated from API');
      }
    } catch (error) {
      console.error('[ChatPage-loadThreads] ❌ Error loading threads:', error);
    } finally {
      setThreadsLoading(false);
      setLoading(false);
    }
  };

  const loadMessagesFromCheckpoint = async (threadId: string) => {
    if (!threadId || !userEmail) {
      console.log('[ChatPage-loadMessages] ❌ Missing threadId or userEmail');
      return;
    }

    console.log('[ChatPage-loadMessages] 🔄 Loading messages for thread:', threadId);

    // If messages are already cached for this thread, use them
    if (activeThreadId === threadId && messages.length > 0) {
      console.log('[ChatPage-loadMessages] ✅ Using cached messages');
      return;
    }

    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      const response = await fetch(`${API_BASE}/chat/${threadId}/messages`, {
        headers: {
          'Authorization': `Bearer ${freshSession.id_token}`
        }
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('[ChatPage-loadMessages] ✅ Loaded messages from API:', data.length);
      
      // Update cache through context
      setMessages(threadId, data);
      setActiveThreadId(threadId);
    } catch (error) {
      console.error('[ChatPage-loadMessages] ❌ Error loading messages:', error);
    }
  };

  const deleteThreadFromPostgreSQL = async (threadId: string) => {
    if (!threadId || !userEmail) {
      console.log('[ChatPage-deleteThread] ❌ Missing threadId or userEmail');
      return false;
    }

    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      const response = await fetch(`${API_BASE}/chat/${threadId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${freshSession.id_token}`
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log('[ChatPage-deleteThread] ✅ Thread deleted successfully');
      
      // Update cache through context
      removeThread(threadId);
      
      // If this was the active thread, clear it
      if (activeThreadId === threadId) {
        setActiveThreadId(null);
      }
      
      return true;
    } catch (error) {
      console.error('[ChatPage-deleteThread] ❌ Error deleting thread:', error);
      return false;
    }
  };

  // Load threads when component mounts or user changes
  useEffect(() => {
    if (userEmail && status === "authenticated") {
      console.log('[ChatPage-useEffect] 🔄 User authenticated, loading threads');
      loadThreadsFromPostgreSQL();
    }
  }, [userEmail, status]);

  // Load messages when active thread changes (when user clicks a thread)
  useEffect(() => {
    if (activeThreadId && threadsLoaded) {
      console.log('[ChatPage-useEffect] 🔄 Active thread changed, loading messages for:', activeThreadId);
      // Save active thread to localStorage
      localStorage.setItem('czsu-last-active-chat', activeThreadId);
      loadMessagesFromCheckpoint(activeThreadId);
    }
  }, [activeThreadId, threadsLoaded]);

  // Restore active thread from localStorage after threads are loaded
  useEffect(() => {
    if (threadsLoaded && threads.length > 0 && !activeThreadId) {
      // Check localStorage for last active thread
      const lastActiveThread = localStorage.getItem('czsu-last-active-chat');
      console.log('[ChatPage-useEffect] 🔄 Checking for last active thread:', lastActiveThread);
      
      if (lastActiveThread && threads.find(t => t.thread_id === lastActiveThread)) {
        console.log('[ChatPage-useEffect] 🔄 Restoring active thread:', lastActiveThread);
        setActiveThreadId(lastActiveThread);
        // Messages will be loaded by the activeThreadId effect above
      } else if (threads.length > 0) {
        // Select the most recent thread if no stored thread
        const mostRecentThread = threads[0]; // threads are sorted by latest_timestamp DESC
        console.log('[ChatPage-useEffect] 🔄 No stored thread, selecting most recent:', mostRecentThread.thread_id);
        setActiveThreadId(mostRecentThread.thread_id);
      }
    }
  }, [threadsLoaded, threads.length, activeThreadId]);

  // Handle URL storage event for localStorage sync
  useEffect(() => {
    function handleStorage(e: StorageEvent) {
      if (e.key === 'czsu-chat-cache' || e.key === 'czsu-last-active-chat') {
        console.log('[ChatPage-storage] 🔄 Storage event detected, potentially reloading');
        // The cache context will handle this automatically
      }
    }

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const handleSQLButtonClick = (msgId: string) => {
    setOpenSQLModalForMsgId(msgId);
  };

  const handleCloseSQLModal = () => {
    setOpenSQLModalForMsgId(null);
  };

  const handleNewChat = async () => {
    if (!userEmail) {
      console.log('[ChatPage-newChat] ❌ No user email available');
      return;
    }

    console.log('[ChatPage-newChat] 🔄 Creating new chat');
    const newThreadId = uuidv4();
    
    const newThread: ChatThreadMeta = {
      thread_id: newThreadId,
      latest_timestamp: new Date().toISOString(),
      run_count: 0,
      title: 'New Chat',
      full_prompt: ''
    };

    // Add to cache through context
    addThread(newThread);
    setActiveThreadId(newThreadId);
    
    // Initialize with empty messages
    setMessages(newThreadId, []);
    
    console.log('[ChatPage-newChat] ✅ New chat created:', newThreadId);
  };

  const handleRename = async (threadId: string, title: string) => {
    console.log('[ChatPage-rename] 🔄 Renaming thread:', threadId, 'to:', title);
    
    // Update cache through context
    updateThread(threadId, { title });
    
    setEditingTitleId(null);
    setNewTitle("");
  };

  const handleDelete = async (threadId: string) => {
    console.log('[ChatPage-delete] 🔄 Deleting thread:', threadId);
    
    if (window.confirm('Are you sure you want to delete this chat thread? This cannot be undone.')) {
      const success = await deleteThreadFromPostgreSQL(threadId);
      if (success) {
        console.log('[ChatPage-delete] ✅ Thread deleted successfully');
      } else {
        console.error('[ChatPage-delete] ❌ Failed to delete thread');
        alert('Failed to delete thread. Please try again.');
      }
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!currentMessage.trim() || isLoading || !userEmail) return;
    
    const messageText = currentMessage.trim();
    setCurrentMessage("");
    setIsLoading(true);
    
    let currentThreadId = activeThreadId;
    
    // Create new thread if none exists
    if (!currentThreadId) {
      currentThreadId = uuidv4();
      const newThread: ChatThreadMeta = {
        thread_id: currentThreadId,
        latest_timestamp: new Date().toISOString(),
        run_count: 0,
        title: messageText.slice(0, 50) + (messageText.length > 50 ? '...' : ''),
        full_prompt: messageText
      };
      
      addThread(newThread);
      setActiveThreadId(currentThreadId);
    }
    
    // Add user message to cache
    const userMessage: ChatMessage = {
      id: uuidv4(),
      threadId: currentThreadId,
      user: userEmail,
      content: messageText,
      isUser: true,
      createdAt: Date.now(),
    };
    
    addMessage(currentThreadId, userMessage);
    
    // Add loading message
    const loadingMessageId = uuidv4();
    const loadingMessage: ChatMessage = {
      id: loadingMessageId,
      threadId: currentThreadId,
      user: 'assistant',
      content: '',
      isUser: false,
      createdAt: Date.now(),
      isLoading: true,
      startedAt: Date.now()
    };
    
    addMessage(currentThreadId, loadingMessage);
    
    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${freshSession.id_token}`
        },
        body: JSON.stringify({
          prompt: messageText,
          thread_id: currentThreadId
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Update loading message with response
      const responseMessage: ChatMessage = {
        id: loadingMessageId,
        threadId: currentThreadId,
        user: 'assistant',
        content: data.result || data.content || 'No response received',
        isUser: false,
        createdAt: Date.now(),
        isLoading: false,
        meta: data.meta || {},
        queriesAndResults: data.queries_and_results || []
      };
      
      updateMessage(currentThreadId, loadingMessageId, responseMessage);
      
      // Update thread metadata and sync localStorage
      updateThread(currentThreadId, {
        latest_timestamp: new Date().toISOString(),
        run_count: (threads.find(t => t.thread_id === currentThreadId)?.run_count || 0) + 1
      });
      
      console.log('[ChatPage-send] ✅ Message sent and localStorage synced with new response');
      
    } catch (error) {
      console.error('[ChatPage-send] ❌ Error sending message:', error);
      
      // Update loading message with error
      const errorMessage: ChatMessage = {
        id: loadingMessageId,
        threadId: currentThreadId,
        user: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
        isUser: false,
        createdAt: Date.now(),
        isLoading: false,
        isError: true,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
      
      updateMessage(currentThreadId, loadingMessageId, errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-scroll when new messages arrive
  React.useEffect(() => {
    if (activeThreadId !== prevChatIdRef.current || messages.length !== prevMsgCountRef.current) {
      setTimeout(() => {
        const messageContainer = document.querySelector('.message-container');
        if (messageContainer) {
          messageContainer.scrollTop = messageContainer.scrollHeight;
        }
      }, 100);
      
      prevChatIdRef.current = activeThreadId;
      prevMsgCountRef.current = messages.length;
    }
  }, [activeThreadId, messages.length]);

  // UI
  return (
    <div className="chat-container flex w-full max-w-7xl mx-auto bg-white rounded-2xl shadow-2xl border border-gray-100 overflow-hidden">
      {/* Sidebar with its own scroll */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
          <span className="font-bold text-lg text-blue-700">Chats</span>
          <button
            className="px-3 py-1.5 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleNewChat}
            title="New chat"
            disabled={isLoading || threads.some(s => !messages.length && s.thread_id === activeThreadId)}
          >
            + New Chat
          </button>
        </div>
        
        {/* Sidebar Chat List with Scroll */}
        <div ref={sidebarRef} className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-1 chat-scrollbar">
          {(threadsLoading && !cacheLoading) ? (
            <div className="text-center py-8">
              <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
              <div className="text-sm text-gray-500">Loading your chats...</div>
            </div>
          ) : threads.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-sm text-gray-500 mb-2">No chats yet</div>
              <div className="text-xs text-gray-400">Click "New Chat" to start</div>
            </div>
          ) : (
            threads.map(s => (
              <div key={s.thread_id} className="group">
                {editingTitleId === s.thread_id ? (
                  <input
                    className="w-full px-3 py-2 text-sm rounded-lg bg-white border border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    value={newTitle}
                    onChange={e => setNewTitle(e.target.value)}
                    onBlur={() => handleRename(s.thread_id, newTitle)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRename(s.thread_id, newTitle); }}
                    autoFocus
                  />
                ) : (
                  <div className="flex items-center min-w-0">
                    <button
                      className={
                        `flex-1 text-left text-sm px-3 py-2 font-semibold rounded-lg transition-all duration-200 cursor-pointer min-w-0 ` +
                        (activeThreadId === s.thread_id
                          ? 'font-extrabold light-blue-theme '
                          : 'text-[#181C3A]/80 hover:text-gray-300 hover:bg-gray-100 ')
                      }
                      style={{fontFamily: 'var(--font-inter)'}}
                      onClick={() => setActiveThreadId(s.thread_id)}
                      onDoubleClick={() => { setEditingTitleId(s.thread_id); setNewTitle(s.title || ''); }}
                      title={`${s.full_prompt || s.title || 'New Chat'}${s.full_prompt && s.full_prompt.length === 50 ? '...' : ''}`}
                    >
                      <div className="truncate block leading-tight">{s.title || 'New Chat'}</div>
                    </button>
                    <button
                      className="flex-shrink-0 ml-1 text-gray-400 hover:text-red-500 text-lg font-bold px-2 py-1 rounded transition-colors"
                      title="Delete chat"
                      onClick={() => handleDelete(s.thread_id)}
                    >
                      ×
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Main Chat Container */}
      <div className="flex-1 flex flex-col bg-gradient-to-br from-white to-blue-50/30 relative">
        {/* Chat Messages Area with its own scroll */}
        <div className="flex-1 overflow-hidden">
          <MessageArea
            messages={messages}
            threadId={activeThreadId}
            onSQLClick={handleSQLButtonClick}
            openSQLModalForMsgId={openSQLModalForMsgId}
            onCloseSQLModal={handleCloseSQLModal}
            onNewChat={handleNewChat}
            isLoading={isLoading}
          />
        </div>
        
        {/* Stationary Input Field */}
        <div className="bg-white border-t border-gray-200 shadow-lg">
          <form onSubmit={handleSend} className="p-4 flex items-center gap-3 max-w-4xl mx-auto">
            <input
              ref={inputRef}
              type="text"
              placeholder="Type your message here..."
              value={currentMessage}
              onChange={e => setCurrentMessage(e.target.value)}
              className="flex-1 px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 bg-gray-50 transition-all duration-200"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="px-6 py-3 light-blue-theme rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              disabled={isLoading || !currentMessage.trim()}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-gray-400 border-t-gray-600 rounded-full animate-spin"></div>
                  Sending...
                </span>
              ) : (
                <span>Send</span>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}