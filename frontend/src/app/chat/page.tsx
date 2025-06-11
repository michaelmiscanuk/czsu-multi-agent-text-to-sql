"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession, getSession, signOut } from "next-auth/react";

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
  
  console.log('[ChatPage-DEBUG] 🔄 Component render - Status:', status, 'UserEmail:', !!userEmail, 'Timestamp:', new Date().toISOString());
  
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
  
  const [threads, setThreads] = useState<ChatThreadMeta[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [openSQLModalForMsgId, setOpenSQLModalForMsgId] = useState<string | null>(null);
  const [iteration, setIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(2); // default fallback
  const [threadsLoaded, setThreadsLoaded] = useState(false);
  const [threadsLoading, setThreadsLoading] = useState(false);
  
  // NEW: Simplified caching state - no force refresh flags needed
  const [threadsCache, setThreadsCache] = useState<ChatThreadMeta[]>([]);
  const [messageCache, setMessageCache] = useState<Record<string, ChatMessage[]>>({});
  const [threadsCacheTimestamp, setThreadsCacheTimestamp] = useState<number>(0);
  const [messageCacheTimestamps, setMessageCacheTimestamps] = useState<Record<string, number>>({});
  
  // NEW: Cache restoration state to prevent race conditions
  const [cacheRestored, setCacheRestored] = useState(false);
  const [isNavigating, setIsNavigating] = useState(false);
  
  // Debug logging for state changes
  React.useEffect(() => {
    console.log('[ChatPage-DEBUG] 📊 State Update - threads:', threads.length, 'activeThreadId:', activeThreadId, 'messages:', messages.length, 'threadsLoaded:', threadsLoaded, 'threadsLoading:', threadsLoading, 'cacheRestored:', cacheRestored);
  }, [threads.length, activeThreadId, messages.length, threadsLoaded, threadsLoading, cacheRestored]);
  
  // Track previous chatId and message count for scroll logic
  const prevChatIdRef = React.useRef<string | null>(null);
  const prevMsgCountRef = React.useRef<number>(1);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const sidebarRef = React.useRef<HTMLDivElement>(null);
  
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

  // Cache duration in milliseconds (longer since we update cache directly)
  const THREADS_CACHE_DURATION = 10 * 60 * 1000; // 10 minutes
  const MESSAGES_CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

  // Helper function to check if cache is still valid
  const isCacheValid = (timestamp: number, duration: number): boolean => {
    return Date.now() - timestamp < duration;
  };

  // Helper functions to update cache directly (chatbot-style)
  const addMessageToCache = (threadId: string, message: ChatMessage) => {
    setMessageCache(prev => ({
      ...prev,
      [threadId]: [...(prev[threadId] || []), message]
    }));
    setMessages(prev => [...prev, message]);
  };

  const updateMessageInCache = (threadId: string, messageId: string, updatedMessage: ChatMessage) => {
    setMessageCache(prev => ({
      ...prev,
      [threadId]: (prev[threadId] || []).map(msg => 
        msg.id === messageId ? updatedMessage : msg
      )
    }));
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? updatedMessage : msg
    ));
  };

  const invalidateMessageCache = (threadId: string) => {
    console.log('[ChatPage-Cache] 🗑️ Invalidating message cache for thread:', threadId);
    setMessageCacheTimestamps(prev => {
      const updated = { ...prev };
      delete updated[threadId]; // Remove timestamp to invalidate cache
      return updated;
    });
  };

  const addThreadToCache = (thread: ChatThreadMeta) => {
    setThreadsCache(prev => [thread, ...prev]);
    setThreads(prev => [thread, ...prev]);
  };

  const removeThreadFromCache = (threadId: string) => {
    setThreadsCache(prev => prev.filter(t => t.thread_id !== threadId));
    setThreads(prev => prev.filter(t => t.thread_id !== threadId));
    // Also remove messages cache for this thread
    setMessageCache(prev => {
      const updated = { ...prev };
      delete updated[threadId];
      return updated;
    });
    setMessageCacheTimestamps(prev => {
      const updated = { ...prev };
      delete updated[threadId];
      return updated;
    });
  };

  const updateThreadInCache = (threadId: string, updates: Partial<ChatThreadMeta>) => {
    setThreadsCache(prev => prev.map(t => 
      t.thread_id === threadId ? { ...t, ...updates } : t
    ));
    setThreads(prev => prev.map(t => 
      t.thread_id === threadId ? { ...t, ...updates } : t
    ));
  };

  // PostgreSQL API functions with simplified caching (chatbot-style)
  const loadThreadsFromPostgreSQL = async () => {
    console.log('[ChatPage-DEBUG] ⏱️ loadThreadsFromPostgreSQL called at:', new Date().toISOString());
    console.log('[ChatPage-PostgreSQL] 🔄 Loading threads from PostgreSQL for user:', userEmail);
    
    if (!userEmail) {
      console.log('[ChatPage-PostgreSQL] ❌ No user email, skipping thread load');
      return;
    }

    // Check cache first - if valid, use it immediately and exit
    const cacheValid = isCacheValid(threadsCacheTimestamp, THREADS_CACHE_DURATION);
    console.log('[ChatPage-DEBUG] 📊 Cache check in loadThreads - timestamp:', threadsCacheTimestamp, 'valid:', cacheValid, 'threadsCache.length:', threadsCache.length);
    
    if (cacheValid && threadsCache.length > 0) {
      console.log('[ChatPage-Cache] 💾 Using cached threads data - instant load');
      setThreads(threadsCache);
      setThreadsLoaded(true);
      setThreadsLoading(false);
      console.log('[ChatPage-DEBUG] ✅ loadThreads completed with cache');
      return; // Exit early - no API call needed
    }
    console.log('[ChatPage-Cache] ⏰ Threads cache expired or empty, fetching fresh data');

    if (threadsLoading) {
      console.log('[ChatPage-PostgreSQL] ⏳ Threads already loading, skipping...');
      return;
    }

    console.log('[ChatPage-DEBUG] 🔄 Setting threadsLoading = true');
    setThreadsLoading(true);

    try {
      console.log('[ChatPage-PostgreSQL] 🔍 Getting fresh session for API call...');
      let freshSession = await getSession();
      
      console.log('[ChatPage-PostgreSQL] 📊 Session debug info:', {
        hasSession: !!freshSession,
        hasIdToken: !!freshSession?.id_token,
        userEmail: freshSession?.user?.email,
        sessionKeys: freshSession ? Object.keys(freshSession) : [],
        tokenPreview: freshSession?.id_token ? freshSession.id_token.slice(0, 50) + '...' : 'none'
      });
      
      if (!freshSession?.id_token) {
        console.log('[ChatPage-PostgreSQL] ❌ No valid session token');
        console.log('[ChatPage-PostgreSQL] 🔍 Session object:', freshSession);
        setThreadsLoaded(true);
        setThreadsLoading(false);
        return;
      }

      console.log('[ChatPage-PostgreSQL] 🔗 Making API call to load threads...');
      console.log('[ChatPage-PostgreSQL] 📍 API URL:', `${API_BASE}/chat-threads`);
      console.log('[ChatPage-PostgreSQL] 🔑 Using token:', freshSession.id_token.slice(0, 50) + '...');
      
      const response = await fetch(`${API_BASE}/chat-threads`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${freshSession.id_token}`
        }
      });

      console.log('[ChatPage-PostgreSQL] 📥 API Response received:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: {
          'content-type': response.headers.get('content-type'),
          'content-length': response.headers.get('content-length')
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[ChatPage-PostgreSQL] ❌ API Error Response:', errorText);
        throw new Error(`Failed to get chat threads: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const postgresThreads = await response.json();
      console.log('[ChatPage-PostgreSQL] ✅ Loaded threads from PostgreSQL:', postgresThreads);
      console.log('[ChatPage-PostgreSQL] 📊 Raw API response details:', {
        isArray: Array.isArray(postgresThreads),
        length: postgresThreads?.length || 0,
        firstItem: postgresThreads?.[0] || null,
        type: typeof postgresThreads
      });
      
      // Convert PostgreSQL format to our frontend format
      const convertedThreads: ChatThreadMeta[] = postgresThreads.map((t: any) => ({
        thread_id: t.thread_id,
        latest_timestamp: t.latest_timestamp,
        run_count: t.run_count,
        title: t.title,
        full_prompt: t.full_prompt
      }));

      console.log('[ChatPage-PostgreSQL] 🔄 Converted threads:', convertedThreads);

      // Update both display state and cache
      setThreads(convertedThreads);
      setThreadsCache(convertedThreads);
      setThreadsCacheTimestamp(Date.now());
      setThreadsLoaded(true);
      
      console.log('[ChatPage-Cache] 💾 Threads cached successfully');
      console.log('[ChatPage-PostgreSQL] 📊 Thread loading summary:');
      console.log(`  - Total threads loaded: ${convertedThreads.length}`);
      console.log(`  - User email: ${userEmail}`);
      console.log(`  - API Base: ${API_BASE}`);
      console.log(`  - Frontend state updated successfully`);
      
      // If no threads exist, automatically create a new one silently
      if (convertedThreads.length === 0) {
        console.log('[ChatPage-PostgreSQL] 📝 No threads found, auto-creating first chat silently');
        const newThreadId = uuidv4();
        
        // Create the thread object and add it to the list
        const firstThread: ChatThreadMeta = {
          thread_id: newThreadId,
          latest_timestamp: new Date().toISOString(),
          run_count: 0,
          title: 'New Chat',
          full_prompt: ''
        };
        
        const newThreadsList = [firstThread];
        setThreads(newThreadsList);
        setThreadsCache(newThreadsList); // Also update cache
        setActiveThreadId(newThreadId);
        setMessages([]); // Clear messages for new chat
        console.log('[ChatPage-PostgreSQL] ✅ Auto-created first thread:', newThreadId);
        console.log('[ChatPage-PostgreSQL] ℹ️ Thread will be created in PostgreSQL when first message is sent');
        
        // Focus input for immediate use
        setTimeout(() => inputRef.current?.focus(), 100);
        setThreadsLoading(false);
        return;
      }
      
      // If no active thread and we have threads, select the most recent one
      if (!activeThreadId && convertedThreads.length > 0) {
        setActiveThreadId(convertedThreads[0].thread_id);
        console.log('[ChatPage-PostgreSQL] 🎯 Auto-selected active thread:', convertedThreads[0].thread_id);
      }
      
    } catch (error) {
      console.error('[ChatPage-PostgreSQL] ❌ Error loading threads from PostgreSQL:', error);
      console.error('[ChatPage-PostgreSQL] 🔍 Error details:', {
        name: (error as Error)?.name || 'Unknown',
        message: (error as Error)?.message || String(error),
        stack: (error as Error)?.stack || 'No stack trace available'
      });
      
      setThreadsLoaded(true); // Still mark as loaded even on error
      
      // If error loading and no threads, create first chat anyway
      if (threads.length === 0) {
        console.log('[ChatPage-PostgreSQL] 📝 Error loading but no threads, auto-creating first chat silently');
        const newThreadId = uuidv4();
        
        // Create the thread object and add it to the list
        const firstThread: ChatThreadMeta = {
          thread_id: newThreadId,
          latest_timestamp: new Date().toISOString(),
          run_count: 0,
          title: 'New Chat',
          full_prompt: ''
        };
        
        const newThreadsList = [firstThread];
        setThreads(newThreadsList);
        setThreadsCache(newThreadsList); // Also update cache
        setActiveThreadId(newThreadId);
        setMessages([]);
        setTimeout(() => inputRef.current?.focus(), 100);
      }
    } finally {
      setThreadsLoading(false);
      console.log('[ChatPage-PostgreSQL] 🏁 Thread loading process completed');
    }
  };

  // Load messages for active session from PostgreSQL checkpoints with simplified caching
  const loadMessagesFromCheckpoint = async (threadId: string) => {
    if (!userEmail || !threadId) {
      setMessages([]);
      return;
    }

    console.log('[ChatPage-PostgreSQL] 📄 Loading COMPLETE messages from checkpoint for thread:', threadId);
    
    // Check cache first
    const cacheTimestamp = messageCacheTimestamps[threadId] || 0;
    const cacheValid = isCacheValid(cacheTimestamp, MESSAGES_CACHE_DURATION);
    const cachedMessages = messageCache[threadId];
    
    if (cacheValid && cachedMessages && cachedMessages.length > 0) {
      console.log('[ChatPage-Cache] 💾 Using cached COMPLETE messages for thread:', threadId, `(${cachedMessages.length} messages)`);
      setMessages(cachedMessages);
      return; // Exit early - no API call needed
    }
    
    // Only show loading behavior if we don't have valid cache
    console.log('[ChatPage-Cache] ⏰ Message cache expired or empty for thread:', threadId, 'fetching COMPLETE conversation history');
    
    try {
      let session = await getSession();
      if (!session?.id_token) {
        console.error('[ChatPage-PostgreSQL] ❌ No session token available');
        setMessages([]);
        return;
      }

      // Fix: Use the correct API_BASE URL instead of /api/
      let response = await fetch(`${API_BASE}/chat/${threadId}/messages`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.id_token}`
        }
      });

      // Handle token refresh if needed
      if (response.status === 401) {
        console.log('[ChatPage-PostgreSQL] 🔄 Token expired during message load, refreshing...');
        const refreshedSession = await update();
        if (!refreshedSession?.id_token) {
          console.error('[ChatPage-PostgreSQL] ❌ Failed to refresh token');
          setMessages([]);
          return;
        }
        
        response = await fetch(`${API_BASE}/chat/${threadId}/messages`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${refreshedSession.id_token}`
          }
        });
      }

      if (!response.ok) {
        console.error('[ChatPage-PostgreSQL] ❌ Failed to load messages:', response.status, response.statusText);
        const errorText = await response.text();
        console.error('[ChatPage-PostgreSQL] ❌ Error response body:', errorText);
        setMessages([]);
        return;
      }

      const chatMessages = await response.json();
      console.log('[ChatPage-PostgreSQL] ✅ Loaded COMPLETE conversation from checkpoint:', chatMessages.length, 'messages');
      
      if (Array.isArray(chatMessages)) {
        // Filter out any temporary loading messages since we now have the complete conversation
        const realMessages = chatMessages.filter(msg => !msg.isLoading && !msg.id.startsWith('temp-'));
        
        console.log('[ChatPage-PostgreSQL] 📋 Complete conversation history loaded:', {
          totalMessages: realMessages.length,
          userMessages: realMessages.filter(m => m.isUser).length,
          aiMessages: realMessages.filter(m => !m.isUser).length
        });
        
        // Update both display state and cache with complete conversation
        setMessages(realMessages);
        setMessageCache(prev => ({
          ...prev,
          [threadId]: realMessages
        }));
        setMessageCacheTimestamps(prev => ({
          ...prev,
          [threadId]: Date.now()
        }));
        console.log('[ChatPage-Cache] 💾 COMPLETE conversation cached for thread:', threadId);
      } else {
        console.error('[ChatPage-PostgreSQL] ❌ Invalid response format - expected array, got:', typeof chatMessages);
        setMessages([]);
      }
      
    } catch (error) {
      console.error('[ChatPage-PostgreSQL] ❌ Error loading COMPLETE conversation from checkpoint:', error);
      setMessages([]);
    }
  };

  const deleteThreadFromPostgreSQL = async (threadId: string) => {
    console.log('[ChatPage-PostgreSQL] 🗑️ Deleting thread from PostgreSQL:', threadId);
    
    try {
      let freshSession = await getSession();
      if (!freshSession?.id_token) {
        console.log('[ChatPage-PostgreSQL] ❌ No valid session token for deletion');
        return false;
      }

      const response = await fetch(`${API_BASE}/chat/${threadId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${freshSession.id_token}`
        }
      });

      if (!response.ok) {
        console.error('[ChatPage-PostgreSQL] ❌ Failed to delete thread:', response.status);
        return false;
      }

      const result = await response.json();
      console.log('[ChatPage-PostgreSQL] ✅ Thread deleted from PostgreSQL:', result);
      return true;
      
    } catch (error) {
      console.error('[ChatPage-PostgreSQL] ❌ Error deleting thread:', error);
      return false;
    }
  };

  // Load threads from PostgreSQL when user email is available (with simplified caching)
  useEffect(() => {
    console.log('[ChatPage-DEBUG] 🔄 Thread loading useEffect triggered - Status:', status, 'UserEmail:', !!userEmail, 'threadsLoaded:', threadsLoaded, 'threadsLoading:', threadsLoading, 'cacheRestored:', cacheRestored);
    
    // Only attempt to load threads when session is authenticated and we have userEmail
    if (status !== "authenticated" || !userEmail) {
      console.log('[ChatPage-PostgreSQL] ⏳ Waiting for authentication... Status:', status, 'UserEmail:', !!userEmail);
      return;
    }
    
    // IMPORTANT: Don't trigger loading if cache was already restored during navigation
    if (cacheRestored) {
      console.log('[ChatPage-PostgreSQL] ✅ Cache already restored, skipping API load');
      return;
    }
    
    // Don't load if already loaded or currently loading
    if (!threadsLoaded && !threadsLoading) {
      console.log('[ChatPage-PostgreSQL] 🚀 Initial thread load triggered - authenticated user:', userEmail);
      loadThreadsFromPostgreSQL();
    } else {
      console.log('[ChatPage-DEBUG] ⏭️ Skipping thread load - threadsLoaded:', threadsLoaded, 'threadsLoading:', threadsLoading);
    }
  }, [status, userEmail, threadsLoaded, threadsLoading, cacheRestored]);

  // Debug: log session changes
  useEffect(() => {
    console.log('[ChatPage-PostgreSQL] 🔍 Session status changed:', {
      status,
      userEmail: userEmail || 'not available',
      threadsLoaded,
      threadsLoading,
      threadsCount: threads.length
    });
  }, [status, userEmail, threadsLoaded, threadsLoading, threads.length]);

  // Load messages for active session (now using PostgreSQL checkpoints with simplified caching)
  useEffect(() => {
    console.log('[ChatPage-DEBUG] 📄 Message loading useEffect triggered - UserEmail:', !!userEmail, 'activeThreadId:', activeThreadId, 'threadsLoaded:', threadsLoaded, 'cacheRestored:', cacheRestored, 'messages.length:', messages.length);
    
    if (!userEmail || !activeThreadId || !threadsLoaded) {
      console.log('[ChatPage-PostgreSQL] ⏳ Not ready to load messages:', {
        userEmail: !!userEmail,
        activeThreadId: !!activeThreadId,
        threadsLoaded
      });
      if (!activeThreadId) {
        console.log('[ChatPage-DEBUG] 🗑️ Clearing messages - no active thread');
        setMessages([]); // Clear messages when no active thread
      }
      return;
    }
    
    // IMPORTANT: If cache was restored and we already have messages, don't reload
    if (cacheRestored && messages.length > 0) {
      console.log('[ChatPage-PostgreSQL] ✅ Messages already restored from cache, skipping API load');
      return;
    }
    
    console.log('[ChatPage-PostgreSQL] 📄 Loading messages for thread:', activeThreadId);
    loadMessagesFromCheckpoint(activeThreadId);
  }, [userEmail, activeThreadId, threadsLoaded, cacheRestored]); // Added cacheRestored dependency

  // Remember last active chat in localStorage
  useEffect(() => {
    if (activeThreadId) {
      localStorage.setItem('czsu-last-active-chat', activeThreadId);
      console.log('[ChatPage-PostgreSQL] 💾 Saved active thread to localStorage:', activeThreadId);
    }
  }, [activeThreadId]);

  // Restore last active chat on mount
  useEffect(() => {
    if (!userEmail || !threadsLoaded) return; // Wait for threads to be loaded
    
    // IMPORTANT: Don't override activeThreadId if cache restoration already set it
    if (cacheRestored && activeThreadId) {
      console.log('[ChatPage-PostgreSQL] ✅ Active thread already restored from cache:', activeThreadId);
      return;
    }
    
    const lastActive = localStorage.getItem('czsu-last-active-chat');
    if (lastActive && threads.length > 0) {
      // Check if the last active thread still exists
      const threadExists = threads.some(t => t.thread_id === lastActive);
      if (threadExists && activeThreadId !== lastActive) { // Only set if different
        console.log('[ChatPage-PostgreSQL] 🔄 Restored active thread from localStorage:', lastActive);
        setActiveThreadId(lastActive);
        // Small delay to ensure proper loading order
        setTimeout(() => {
          console.log('[ChatPage-PostgreSQL] 📄 Triggering message reload after thread restoration');
        }, 100);
      } else if (!threadExists) {
        console.log('[ChatPage-PostgreSQL] ⚠️ Last active thread no longer exists, clearing localStorage');
        localStorage.removeItem('czsu-last-active-chat');
        // Select first available thread
        if (!activeThreadId && threads.length > 0) {
          setActiveThreadId(threads[0].thread_id);
          console.log('[ChatPage-PostgreSQL] 🎯 Auto-selected first available thread after invalid restore:', threads[0].thread_id);
        }
      }
    } else if (!activeThreadId && threads.length > 0) {
      setActiveThreadId(threads[0].thread_id);
      console.log('[ChatPage-PostgreSQL] 🎯 Auto-selected first available thread:', threads[0].thread_id);
    }
  }, [userEmail, threads.length, threadsLoaded, cacheRestored]); // Added cacheRestored dependency

  // Auto-scroll sidebar to top when entering chat menu
  useEffect(() => {
    if (sidebarRef.current) {
      sidebarRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

  // Auto-create first chat if none exist (after threads are loaded)
  useEffect(() => {
    if (!userEmail || !threadsLoaded) return;
    
    if (threads.length === 0) {
      console.log('[ChatPage-PostgreSQL] 🆕 No threads found, will create one on first message');
      // We'll create the thread automatically when the user sends their first message
    }
  }, [userEmail, threadsLoaded, threads.length]);

  // Clear caches when user changes (reset everything for new user)
  useEffect(() => {
    if (userEmail) {
      console.log('[ChatPage-Cache] 👤 User changed, clearing all caches');
      setThreadsCacheTimestamp(0);
      setMessageCacheTimestamps({});
      setThreadsCache([]);
      setMessageCache({});
    }
  }, [userEmail]);

  // Smart cache invalidation: Only clear on actual page refresh, not on navigation
  useEffect(() => {
    console.log('[ChatPage-DEBUG] 🚀 Cache invalidation useEffect triggered - Start timestamp:', new Date().toISOString());
    
    const isPageRefresh = () => {
      console.log('[ChatPage-DEBUG] 🔍 Checking if page refresh...');
      
      // Method 1: Check if this is the first render after a hard refresh
      // Use a combination of sessionStorage and performance timing
      const sessionKey = 'czsu-chat-session-active';
      const pageLoadKey = 'czsu-chat-page-load-time';
      const currentTime = Date.now();
      
      // Check if session storage exists (survives navigation but not refresh)
      const sessionActive = sessionStorage.getItem(sessionKey);
      const lastPageLoad = sessionStorage.getItem(pageLoadKey);
      
      console.log('[ChatPage-DEBUG] 📊 Session check - sessionActive:', !!sessionActive, 'lastPageLoad:', lastPageLoad);
      
      // If no session marker, this is likely a fresh page load (F5, new tab, direct URL)
      if (!sessionActive) {
        sessionStorage.setItem(sessionKey, 'true');
        sessionStorage.setItem(pageLoadKey, currentTime.toString());
        console.log('[ChatPage-DEBUG] 📊 No session marker - treating as page refresh');
        return true;
      }
      
      // If session exists, check if this is a very recent page load (within 1 second)
      // This catches F5 refreshes where sessionStorage might persist briefly
      if (lastPageLoad) {
        const timeSinceLoad = currentTime - parseInt(lastPageLoad);
        console.log('[ChatPage-DEBUG] 📊 Time since last load:', timeSinceLoad, 'ms');
        
        if (timeSinceLoad < 1000) { // Less than 1 second = likely a refresh
          sessionStorage.setItem(pageLoadKey, currentTime.toString());
          console.log('[ChatPage-DEBUG] 📊 Recent page load detected - treating as page refresh');
          return true;
        }
      }
      
      // Method 2: Check navigation timing as secondary indicator
      if (typeof window !== 'undefined' && window.performance) {
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        if (navigation) {
          console.log('[ChatPage-DEBUG] 📊 Navigation type detected:', navigation.type);
          
          // Only trust 'reload' type if we don't have session markers indicating navigation
          if (navigation.type === 'reload' && !sessionActive) {
            console.log('[ChatPage-DEBUG] 📊 Navigation API confirms page refresh');
            return true;
          }
        }
      }
      
      // Method 3: Check document.referrer for navigation context
      if (typeof document !== 'undefined' && document.referrer) {
        const referrer = new URL(document.referrer);
        const current = new URL(window.location.href);
        
        // If referrer is from the same origin and different path, it's likely navigation
        if (referrer.origin === current.origin && referrer.pathname !== current.pathname) {
          console.log('[ChatPage-DEBUG] 📊 Same-origin navigation detected from:', referrer.pathname, 'to:', current.pathname);
          sessionStorage.setItem(pageLoadKey, currentTime.toString());
          return false; // This is navigation, not refresh
        }
      }
      
      // Default: if we have session markers, treat as navigation
      console.log('[ChatPage-DEBUG] 📊 Session markers exist - treating as navigation');
      sessionStorage.setItem(pageLoadKey, currentTime.toString());
      return false;
    };

    const shouldClearCache = isPageRefresh();
    console.log('[ChatPage-DEBUG] 🎯 Cache decision - shouldClearCache:', shouldClearCache);
    
    if (shouldClearCache) {
      console.log('[ChatPage-Cache] 🔄 Actual page refresh detected - invalidating caches for fresh data');
      console.log('[ChatPage-DEBUG] 🗑️ Clearing all cache states...');
      setThreadsCacheTimestamp(0);
      setMessageCacheTimestamps({});
      setThreadsCache([]);
      setMessageCache({});
      setThreadsLoaded(false); // Reset loading state for fresh data
      setCacheRestored(false);
      setIsNavigating(false);
      console.log('[ChatPage-DEBUG] ✅ Cache cleared for fresh data');
    } else {
      console.log('[ChatPage-Cache] 🚀 Navigation detected - IMMEDIATE cache restoration');
      console.log('[ChatPage-DEBUG] ⏱️ Starting navigation cache restoration at:', new Date().toISOString());
      setIsNavigating(true);
      
      // IMMEDIATE: Check if we have valid cached data
      const cacheValid = isCacheValid(threadsCacheTimestamp, THREADS_CACHE_DURATION);
      console.log('[ChatPage-DEBUG] 📊 Cache validation - timestamp:', threadsCacheTimestamp, 'valid:', cacheValid, 'threadsCache.length:', threadsCache.length);
      
      if (cacheValid && threadsCache.length > 0) {
        console.log('[ChatPage-Cache] ⚡ INSTANT cache restoration - no loading states allowed');
        console.log('[ChatPage-DEBUG] 🔄 Restoring threads cache:', threadsCache.length, 'threads');
        
        // Set ALL states immediately in a single batch to prevent any loading UI
        setThreads(threadsCache);
        setThreadsLoaded(true);
        setThreadsLoading(false);
        setCacheRestored(true);
        console.log('[ChatPage-DEBUG] ✅ Basic states restored');
        
        // Immediately restore active thread and messages
        const lastActive = localStorage.getItem('czsu-last-active-chat');
        console.log('[ChatPage-DEBUG] 📊 LastActive from localStorage:', lastActive);
        
        if (lastActive && threadsCache.some(t => t.thread_id === lastActive)) {
          console.log('[ChatPage-DEBUG] 🎯 Restoring active thread:', lastActive);
          setActiveThreadId(lastActive);
          
          // Also immediately restore messages if cached
          const msgCacheTimestamp = messageCacheTimestamps[lastActive] || 0;
          const msgCacheValid = isCacheValid(msgCacheTimestamp, MESSAGES_CACHE_DURATION);
          const cachedMessages = messageCache[lastActive];
          console.log('[ChatPage-DEBUG] 📊 Message cache check - timestamp:', msgCacheTimestamp, 'valid:', msgCacheValid, 'messages:', cachedMessages?.length || 0);
          
          if (msgCacheValid && cachedMessages) {
            console.log('[ChatPage-Cache] ⚡ INSTANT message restoration for thread:', lastActive);
            setMessages(cachedMessages);
            console.log('[ChatPage-DEBUG] ✅ Messages restored:', cachedMessages.length);
          }
          
          console.log('[ChatPage-Cache] ✅ Complete instant restoration - thread and messages');
        } else if (threadsCache.length > 0) {
          const firstThread = threadsCache[0].thread_id;
          console.log('[ChatPage-DEBUG] 🎯 Setting first thread as active:', firstThread);
          setActiveThreadId(firstThread);
          
          // Also restore messages for first thread if available
          const msgCacheTimestamp = messageCacheTimestamps[firstThread] || 0;
          const msgCacheValid = isCacheValid(msgCacheTimestamp, MESSAGES_CACHE_DURATION);
          const cachedMessages = messageCache[firstThread];
          console.log('[ChatPage-DEBUG] 📊 First thread message cache check - valid:', msgCacheValid, 'messages:', cachedMessages?.length || 0);
          
          if (msgCacheValid && cachedMessages) {
            console.log('[ChatPage-Cache] ⚡ INSTANT message restoration for first thread:', firstThread);
            setMessages(cachedMessages);
            console.log('[ChatPage-DEBUG] ✅ First thread messages restored:', cachedMessages.length);
          }
          
          console.log('[ChatPage-Cache] ✅ Set first thread as active with instant restoration');
        }
      } else {
        console.log('[ChatPage-Cache] ⚠️ No valid cache available for navigation');
        console.log('[ChatPage-DEBUG] 📊 Cache invalid - timestamp:', threadsCacheTimestamp, 'valid:', cacheValid, 'threadsCache.length:', threadsCache.length);
        setCacheRestored(false);
      }
      
      setIsNavigating(false);
      console.log('[ChatPage-DEBUG] ⏱️ Navigation cache restoration completed at:', new Date().toISOString());
    }
    
    console.log('[ChatPage-DEBUG] 🏁 Cache invalidation useEffect completed');
  }, []); // Empty dependency array = runs only on component mount

  // Debug: log session on mount and when it changes
  useEffect(() => {
    console.log('[ChatPage-PostgreSQL] 👤 Session updated:', JSON.stringify(session, null, 2));
  }, [session]);

  // Sync isLoading across tabs/windows
  useEffect(() => {
    function handleStorage(e: StorageEvent) {
      if (e.key === 'czsu-chat-isLoading') {
        setIsLoading(e.newValue === 'true');
      }
    }
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  useEffect(() => {
    if (isLoading) {
      localStorage.setItem('czsu-chat-isLoading', 'true');
    } else {
      localStorage.setItem('czsu-chat-isLoading', 'false');
    }
  }, [isLoading]);

  const handleSQLButtonClick = (msgId: string) => {
    setOpenSQLModalForMsgId(msgId);
  };

  const handleCloseSQLModal = () => {
    setOpenSQLModalForMsgId(null);
  };

  // New chat
  const handleNewChat = async () => {
    if (!userEmail) return;
    
    // Prevent creating multiple empty chats
    const hasEmptyChat = threads.some(t => t.run_count === 0 || !t.run_count);
    if (hasEmptyChat) {
      console.log('[ChatPage-PostgreSQL] ⚠️ Empty chat already exists, not creating another');
      return;
    }
    
    console.log('[ChatPage-PostgreSQL] ➕ Creating new chat...');
    const newThreadId = uuidv4();
    
    // Create a new thread entry in the threads list
    const newThread: ChatThreadMeta = {
      thread_id: newThreadId,
      latest_timestamp: new Date().toISOString(),
      run_count: 0,
      title: 'New Chat',
      full_prompt: ''
    };
    
    // Add to cache directly (chatbot-style)
    addThreadToCache(newThread);
    setActiveThreadId(newThreadId);
    setMessages([]); // Clear messages for new chat
    
    console.log('[ChatPage-PostgreSQL] ✅ New thread created and added to sidebar:', newThreadId);
    
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  // Rename chat - we'll implement this later as it needs thread metadata storage
  const handleRename = async (threadId: string, title: string) => {
    console.log('[ChatPage-PostgreSQL] ✏️ Chat renaming not yet implemented for PostgreSQL backend');
    // TODO: Implement thread title storage in PostgreSQL
    setEditingTitleId(null);
  };

  // Delete chat
  const handleDelete = async (threadId: string) => {
    if (!userEmail) return;
    
    console.log('[ChatPage-PostgreSQL] 🗑️ Deleting chat:', threadId);
    
    try {
      const success = await deleteThreadFromPostgreSQL(threadId);
      
      if (success) {
        console.log('[ChatPage-PostgreSQL] ✅ Thread deleted successfully');
        
        // Update cache directly (chatbot-style)
        removeThreadFromCache(threadId);
        
        // If we deleted the active thread, switch to another one
        if (activeThreadId === threadId) {
          const remainingThreads = threads.filter(t => t.thread_id !== threadId);
          const newActiveThread = remainingThreads.length > 0 ? remainingThreads[0].thread_id : null;
          setActiveThreadId(newActiveThread);
          console.log('[ChatPage-PostgreSQL] 🎯 Switched to new active thread:', newActiveThread);
        }
      } else {
        console.error('[ChatPage-PostgreSQL] ❌ Failed to delete thread');
      }
    } catch (error) {
      console.error('[ChatPage-PostgreSQL] ❌ Error deleting chat:', error);
    }
  };

  // Send message
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userEmail || !currentMessage.trim()) return;
    
    console.log('[ChatPage-PostgreSQL] 📤 Sending message to existing conversation thread:', currentMessage.slice(0, 50) + '...');
    setIsLoading(true);
    
    let threadId = activeThreadId;
    
    // If no active thread OR no threads exist at all, create a new one
    // This ensures we always have a parent chat item for any question
    if (!threadId || threads.length === 0) {
      threadId = uuidv4();
      setActiveThreadId(threadId);
      console.log('[ChatPage-PostgreSQL] 🆕 Auto-created thread for message (ensuring parent chat item):', threadId);
    }

    // Store the current message and clear input immediately
    const userMessageContent = currentMessage;
    setCurrentMessage("");
    
    // Add optimistic UI updates - show user message and loading immediately
    const tempUserMessage: ChatMessage = {
      id: `temp-user-${Date.now()}`,
      threadId: threadId,
      user: userEmail,
      content: userMessageContent,
      isUser: true,
      createdAt: Date.now()
    };
    
    const tempLoadingMessage: ChatMessage = {
      id: `temp-ai-${Date.now()}`,
      threadId: threadId,
      user: 'AI',
      content: '',
      isUser: false,
      createdAt: Date.now(),
      isLoading: true,
      startedAt: Date.now()
    };
    
    // Append to existing conversation (don't replace)
    setMessages(prev => [...prev, tempUserMessage, tempLoadingMessage]);
    console.log('[ChatPage-PostgreSQL] 💬 Added user message and loading state to existing conversation');

    try {
      let freshSession = await getSession();
      if (!freshSession?.id_token) {
        signOut();
        setIsLoading(false);
        return;
      }
      
      const API_URL = `${API_BASE}/analyze`;
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${freshSession.id_token}`
      };
      
      console.log('[ChatPage-PostgreSQL] 🚀 Calling analyze API for follow-up message with thread_id:', threadId);
      
      let response = await fetch(API_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify({ prompt: userMessageContent, thread_id: threadId })
      });
      
      // Handle token refresh if needed
      if (response.status === 401) {
        console.log('[ChatPage-PostgreSQL] 🔄 Token expired, refreshing...');
        const refreshedSession = await update();
        if (!refreshedSession?.id_token) {
          signOut();
          setIsLoading(false);
          return;
        }
        const retryHeaders = {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${refreshedSession.id_token}`
        };
        response = await fetch(API_URL, {
          method: 'POST',
          headers: retryHeaders,
          body: JSON.stringify({ prompt: userMessageContent, thread_id: threadId })
        });
        if (response.status === 401) {
          signOut();
          setIsLoading(false);
          return;
        }
      }
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('[ChatPage-PostgreSQL] ✅ Received response, run_id:', data.run_id);
      
      // IMMEDIATE: Display the final answer from backend response without delay
      console.log('[ChatPage-PostgreSQL] 🎯 Displaying final answer immediately from backend response');
      
      // Remove the temporary loading message and add the real AI response
      setMessages(prev => {
        // Filter out temporary messages
        const withoutTemp = prev.filter(msg => !msg.id.startsWith('temp-ai-'));
        
        // Add the real AI response with data from backend
        const aiResponse: ChatMessage = {
          id: `ai-${Date.now()}`,
          threadId: threadId,
          user: 'AI',
          content: data.result || 'No response received',
          isUser: false,
          createdAt: Date.now(),
          isLoading: false,
          meta: {
            datasetsUsed: data.top_selection_codes || [],
            sqlQuery: data.sql || null,
            datasetUrl: data.datasetUrl || null,
            run_id: data.run_id // Store run_id for feedback
          },
          queriesAndResults: data.queries_and_results || []
        };
        
        return [...withoutTemp, aiResponse];
      });
      
      console.log('[ChatPage-PostgreSQL] ✅ Final answer displayed immediately to user');
      
      // BACKGROUND: Refresh caches for consistency (don't make user wait)
      console.log('[ChatPage-PostgreSQL] 🔄 Refreshing caches in background for consistency...');
      
      // Clear caches and refresh in background (without await to not block UI)
      Promise.resolve().then(async () => {
        try {
          // Small delay to let checkpoint settle
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // Clear message cache for this thread
          setMessageCache(prev => {
            const updated = { ...prev };
            delete updated[threadId];
            return updated;
          });
          setMessageCacheTimestamps(prev => {
            const updated = { ...prev };
            delete updated[threadId];
            return updated;
          });
          
          // Refresh thread cache to update message counts and titles
          setThreadsCacheTimestamp(0);
          await loadThreadsFromPostgreSQL();
          
          console.log('[ChatPage-PostgreSQL] ✅ Background cache refresh completed');
        } catch (error) {
          console.error('[ChatPage-PostgreSQL] ⚠ Background cache refresh failed:', error);
        }
      });
      
      // Show success message
      console.log('[ChatPage-PostgreSQL] 🎉 Message sent successfully - answer displayed immediately');
      
    } catch (error) {
      console.error('[ChatPage-PostgreSQL] ❌ Error sending message:', error);
      
      // Remove temporary loading message and show error immediately
      setMessages(prev => {
        // Filter out temporary loading messages
        const withoutTempLoading = prev.filter(msg => !msg.id.startsWith('temp-ai-'));
        
        // Add error message
        const errorMessage: ChatMessage = {
          id: `error-${Date.now()}`,
          threadId: threadId,
          user: 'System',
          content: `Error: ${error}`,
          isUser: false,
          createdAt: Date.now(),
          isLoading: false,
          isError: true
        };
        
        return [...withoutTempLoading, errorMessage];
      });
      
      // Background cache cleanup (don't block UI)
      Promise.resolve().then(async () => {
        try {
          setMessageCache(prev => {
            const updated = { ...prev };
            delete updated[threadId];
            return updated;
          });
          setMessageCacheTimestamps(prev => {
            const updated = { ...prev };
            delete updated[threadId];
            return updated;
          });
          console.log('[ChatPage-PostgreSQL] ✅ Background error cleanup completed');
        } catch (cleanupError) {
          console.error('[ChatPage-PostgreSQL] ⚠ Background cleanup failed:', cleanupError);
        }
      });
    } finally {
      setIsLoading(false);
    }
  };

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
          {(threadsLoading && !cacheRestored) ? (
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