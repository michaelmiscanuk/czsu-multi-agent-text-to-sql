"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession, getSession, signOut } from "next-auth/react";
import { useChatCache } from '@/contexts/ChatCacheContext';
import { ChatThreadMeta, ChatMessage, AnalyzeResponse, ChatThreadResponse } from '@/types';
import { API_CONFIG, authApiFetch } from '@/lib/api';

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
    forceAPIRefresh,
    hasMessagesForThread,
    resetPageRefresh,
    isUserLoading,
    setUserLoadingState,
    checkUserLoadingState,
    setUserEmail
  } = useChatCache();
  
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
  
  // Combined loading state: local loading OR global context loading OR cross-tab user loading
  // This ensures loading state persists across navigation AND across browser tabs for the same user
  const isAnyLoading = isLoading || cacheLoading || isUserLoading;
  
  // Track previous chatId and message count for scroll logic
  const prevChatIdRef = React.useRef<string | null>(null);
  const prevMsgCountRef = React.useRef<number>(1);
  const inputRef = React.useRef<HTMLTextAreaElement>(null);
  const sidebarRef = React.useRef<HTMLDivElement>(null);
  
  // PostgreSQL API functions with new cache context
  const loadThreadsFromPostgreSQL = async () => {
    if (!userEmail) {
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
      const data = await authApiFetch<ChatThreadResponse[]>('/chat-threads', freshSession.id_token);
      
      console.log('[ChatPage-loadThreads] ✅ Loaded threads from PostgreSQL API:', data.length);
      
      // Update cache through context - this will sync localStorage with PostgreSQL data
      setThreads(data);
      setThreadsLoaded(true);
      
      if (isPageRefresh) {
        console.log('[ChatPage-loadThreads] ✅ F5 refresh completed - localStorage now synced with PostgreSQL');
        // IMPORTANT: Reset page refresh flag after successful F5 sync
        // This allows subsequent navigation to use cache instead of always hitting API
        resetPageRefresh();
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
      return;
    }

    console.log('[ChatPage-loadMessages] 🔄 Loading messages for thread:', threadId);

    // CHECK CACHE FIRST: Use the new context method to check if messages exist for this thread
    const hasCachedMessages = hasMessagesForThread(threadId);
    
    // Use cached messages if available AND it's not a page refresh
    if (hasCachedMessages && !isPageRefresh) {
      console.log('[ChatPage-loadMessages] ✅ Using cached messages for thread:', threadId);
      setActiveThreadId(threadId);
      return;
    }

    // Only make API call if no cached messages OR it's a page refresh (to sync with PostgreSQL)
    console.log('[ChatPage-loadMessages] 📡 Making API call for thread:', threadId, 'reason:', !hasCachedMessages ? 'no cache' : 'page refresh');

    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      const data = await authApiFetch<ChatMessage[]>(`/chat/${threadId}/messages`, freshSession.id_token);
      
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
      return false;
    }
    
    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      await authApiFetch(`/chat/${threadId}`, freshSession.id_token, {
        method: 'DELETE',
      });

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

  // NEW: Initialize user email in context and check for existing loading state
  useEffect(() => {
    if (userEmail && status === "authenticated") {
      // Set user email in context for localStorage operations
      console.log('[ChatPage-useEffect] 👤 Setting user email in context:', userEmail);
      setUserEmail(userEmail);
      
      // Check if user already has a loading state from another tab
      const existingLoadingState = checkUserLoadingState(userEmail);
      if (existingLoadingState) {
        console.log('[ChatPage-useEffect] 🔒 Found existing loading state for user:', userEmail, '- blocking new requests');
      }
    }
  }, [userEmail, status, checkUserLoadingState, setUserEmail]);

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

  // NEW: Cleanup effect for cross-tab loading state
  useEffect(() => {
    // Cleanup function when user changes or component unmounts
    return () => {
      if (userEmail) {
        console.log('[ChatPage-cleanup] 🧹 Component unmounting, checking if we should clean up loading state for:', userEmail);
        // Only clean up if this tab/component is actually loading
        if (isLoading && userEmail) {
          setUserLoadingState(userEmail, false);
          console.log('[ChatPage-cleanup] ✅ Cleaned up loading state for:', userEmail);
        }
      }
    };
  }, [userEmail, isLoading, setUserLoadingState]);

  const handleSQLButtonClick = (msgId: string) => {
    setOpenSQLModalForMsgId(msgId);
  };

  const handleCloseSQLModal = () => {
    setOpenSQLModalForMsgId(null);
  };

  const handleNewChat = async () => {
    if (!userEmail) {
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
    
    if (!currentMessage.trim() || isAnyLoading || !userEmail) return;
    
    // CRITICAL: Block if user is already loading in ANY tab before doing anything else
    const existingLoadingState = checkUserLoadingState(userEmail);
    if (existingLoadingState) {
      console.log('[ChatPage-send] 🚫 BLOCKED: User', userEmail, 'is already processing a request in another tab');
      return; // Exit immediately - don't allow concurrent requests
    }
    
    console.log('[ChatPage-send] ✅ No existing loading state found, proceeding with request for user:', userEmail);
    
    const messageText = currentMessage.trim();
    setCurrentMessage("");
    
    // Set loading state in BOTH local and context to ensure persistence across navigation
    setIsLoading(true);
    setLoading(true); // Context loading state - persists across navigation
    
    // NEW: Set cross-tab loading state tied to user email IMMEDIATELY
    setUserLoadingState(userEmail, true);
    
    let currentThreadId = activeThreadId;
    let shouldUpdateTitle = false;
    
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
      console.log('[ChatPage-send] ✅ Created new thread with title:', newThread.title);
    } else {
      // Check if existing thread needs title update
      const currentThread = threads.find(t => t.thread_id === currentThreadId);
      const currentMessages = messages.filter(msg => msg.isUser); // Count only user messages
      
      // Update title if:
      // 1. Current title is generic ("New Chat" or empty)
      // 2. OR this is the first user message in the thread
      // 3. OR the full_prompt is empty/missing
      if (currentThread && 
          (currentThread.title === 'New Chat' || !currentThread.title || 
           currentMessages.length === 0 || !currentThread.full_prompt)) {
        shouldUpdateTitle = true;
        const newTitle = messageText.slice(0, 50) + (messageText.length > 50 ? '...' : '');
        
        // IMMEDIATELY update thread title in frontend state and localStorage
        updateThread(currentThreadId, {
          title: newTitle,
          full_prompt: messageText,
          latest_timestamp: new Date().toISOString()
        });
        
        console.log('[ChatPage-send] ✅ Immediately updated existing thread title to:', newTitle);
      }
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

      const data = await authApiFetch<AnalyzeResponse>('/analyze', freshSession.id_token, {
        method: 'POST',
        body: JSON.stringify({
          prompt: messageText,
          thread_id: currentThreadId
        }),
      });
      
      console.log('[ChatPage-send] ✅ Response received with run_id:', data.run_id);
      
      // Update loading message with response
      const responseMessage: ChatMessage = {
        id: loadingMessageId,
        threadId: currentThreadId,
        user: 'assistant',
        content: data.result || 'No response received',
        isUser: false,
        createdAt: Date.now(),
        isLoading: false,
        meta: {
          datasetsUsed: data.top_selection_codes || [],
          sqlQuery: data.sql || null,
          datasetUrl: data.datasetUrl || null,
          run_id: data.run_id  // Store run_id in meta for feedback
        },
        queriesAndResults: data.queries_and_results || []
      };
      
      console.log('[ChatPage-send] ✅ Attaching run_id to message meta:', data.run_id);
      
      updateMessage(currentThreadId, loadingMessageId, responseMessage);
      
      // Update thread metadata after response - this ensures any PostgreSQL changes are reflected
      const updatedMetadata: Partial<ChatThreadMeta> = {
        latest_timestamp: new Date().toISOString(),
        run_count: (threads.find(t => t.thread_id === currentThreadId)?.run_count || 0) + 1
      };
      
      // If we updated the title earlier, make sure to preserve it in case PostgreSQL response overwrites
      if (shouldUpdateTitle) {
        const newTitle = messageText.slice(0, 50) + (messageText.length > 50 ? '...' : '');
        updatedMetadata.title = newTitle;
        updatedMetadata.full_prompt = messageText;
        console.log('[ChatPage-send] ✅ Preserving updated title after response:', newTitle);
      }
      
      updateThread(currentThreadId, updatedMetadata);
      
      console.log('[ChatPage-send] ✅ Message sent and localStorage synced with new response');
      
      // IMPORTANT: After receiving response from langgraph, reload messages from PostgreSQL to ensure sync
      console.log('[ChatPage-send] 🔄 Reloading messages from PostgreSQL to sync with langgraph response');
      try {
        const freshSession = await getSession();
        if (freshSession?.id_token) {
          const freshMessages = await authApiFetch<ChatMessage[]>(`/chat/${currentThreadId}/messages`, freshSession.id_token);
          console.log('[ChatPage-send] ✅ Reloaded', freshMessages.length, 'messages from PostgreSQL after langgraph response');
          setMessages(currentThreadId, freshMessages);
        }
      } catch (refreshError) {
        console.error('[ChatPage-send] ⚠️ Failed to refresh messages from PostgreSQL:', refreshError);
        // Don't fail the whole operation if refresh fails
      }
    } catch (error) {
      console.error('[ChatPage-send] ❌ Error sending message:', error);
      console.error('[ChatPage-send] ❌ Error details:', {
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : 'No stack trace',
        type: typeof error,
        error: error
      });
      
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
      setLoading(false); // Clear context loading state as well
      
      // NEW: Clear cross-tab loading state tied to user email
      setUserLoadingState(userEmail, false);
    }
  };

  // Add keyboard handler for SHIFT+ENTER vs ENTER
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Only submit if there's content and not loading
      if (currentMessage.trim() && !isAnyLoading) {
        const syntheticEvent = new Event('submit') as any;
        syntheticEvent.preventDefault = () => {};
        handleSend(syntheticEvent);
      }
    }
    // SHIFT+ENTER will naturally create a new line due to default textarea behavior
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

  // Auto-resize textarea when currentMessage changes programmatically
  React.useEffect(() => {
    if (inputRef.current) {
      const textarea = inputRef.current;
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  }, [currentMessage]);

  // Focus input field when activeThreadId changes, threads are loaded, or component mounts
  React.useEffect(() => {
    const focusInput = () => {
      if (inputRef.current) {
        setTimeout(() => {
          inputRef.current?.focus();
        }, 100);
      }
    };

    // Focus when activeThreadId changes (switching chats or creating new chat)
    if (activeThreadId) {
      focusInput();
    }
    // Focus when threads are loaded for the first time
    else if (threadsLoaded && threads.length === 0) {
      focusInput();
    }
    // Focus on initial mount when no active thread
    else if (!activeThreadId && !cacheLoading && !threadsLoading) {
      focusInput();
    }
  }, [activeThreadId, threadsLoaded, threads.length, cacheLoading, threadsLoading]);

  // Focus input field when navigating back to chat page
  React.useEffect(() => {
    const focusInput = () => {
      if (inputRef.current) {
        setTimeout(() => {
          inputRef.current?.focus();
        }, 200);
      }
    };

    // Focus when component mounts (page navigation)
    focusInput();
  }, []);

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
            disabled={isAnyLoading || threads.some(s => !messages.length && s.thread_id === activeThreadId)}
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
            isLoading={isAnyLoading}
            isAnyLoading={isAnyLoading}
            threads={threads}
            activeThreadId={activeThreadId}
          />
        </div>
        
        {/* Stationary Input Field */}
        <div className="bg-white border-t border-gray-200 shadow-lg">
          <form onSubmit={handleSend} className="p-4 flex items-start gap-3 max-w-4xl mx-auto">
            <textarea
              ref={inputRef}
              placeholder="Type your message here... (SHIFT+ENTER for new line)"
              value={currentMessage}
              onChange={e => setCurrentMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 bg-gray-50 transition-all duration-200 resize-none min-h-[48px] max-h-[200px]"
              disabled={isAnyLoading}
              rows={1}
              style={{
                height: 'auto',
                minHeight: '48px',
                maxHeight: '200px'
              }}
              onInput={(e) => {
                // Auto-resize textarea based on content
                const target = e.target as HTMLTextAreaElement;
                target.style.height = 'auto';
                target.style.height = Math.min(target.scrollHeight, 200) + 'px';
              }}
            />
            <button
              type="submit"
              className="px-6 py-3 light-blue-theme rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 mt-1"
              disabled={isAnyLoading || !currentMessage.trim()}
            >
              {isAnyLoading ? (
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