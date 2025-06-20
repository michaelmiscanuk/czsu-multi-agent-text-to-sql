"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession, getSession, signOut } from "next-auth/react";
import { useChatCache } from '@/contexts/ChatCacheContext';
import { ChatThreadMeta, ChatMessage, AnalyzeResponse, ChatThreadResponse } from '@/types';
import { API_CONFIG, authApiFetch } from '@/lib/api';
import { useInfiniteScroll } from '@/lib/hooks/useInfiniteScroll';
import LoadingSpinner from '@/components/LoadingSpinner';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  startedAt?: number;
  isError?: boolean;
  selectionCode?: string | null;
  queriesAndResults?: [string, string][];
  meta?: {
    datasetUrl?: string;
    datasetsUsed?: string[];  // Array of dataset codes actually used in queries
    sqlQuery?: string;
    run_id?: string;
    topChunks?: Array<{
      content: string;
      metadata: Record<string, any>;
    }>;
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
    threadsPage,
    threadsHasMore,
    threadsLoading,
    totalThreadsCount,
    setThreads,
    setMessages,
    setActiveThreadId,
    addMessage,
    updateMessage,
    addThread,
    removeThread,
    updateThread,
    loadInitialThreads,
    loadMoreThreads,
    resetPagination,
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
    setUserEmail,
    clearCacheForUserChange,
    loadAllMessagesFromAPI,
    getRunIdsForThread,
    getSentimentsForThread,
    updateCachedSentiment
  } = useChatCache();
  
  // Infinite scroll for threads
  const {
    isLoading: infiniteScrollLoading,
    error: infiniteScrollError,
    hasMore: infiniteScrollHasMore,
    observerRef: threadsObserverRef,
    setHasMore: setInfiniteScrollHasMore,
    setError: setInfiniteScrollError
  } = useInfiniteScroll(
    async () => {
      await loadMoreThreads();
    },
    { threshold: 1.0, rootMargin: '100px' }
  );

  // Update infinite scroll state when pagination state changes
  useEffect(() => {
    setInfiniteScrollHasMore(threadsHasMore);
  }, [threadsHasMore, setInfiniteScrollHasMore]);

  // Clear infinite scroll error when needed
  useEffect(() => {
    if (infiniteScrollError) {
      console.error('[ChatPage] Infinite scroll error:', infiniteScrollError);
      setInfiniteScrollError(null);
    }
  }, [infiniteScrollError, setInfiniteScrollError]);

  // Local component state
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [openSQLModalForMsgId, setOpenSQLModalForMsgId] = useState<string | null>(null);
  const [openPDFModalForMsgId, setOpenPDFModalForMsgId] = useState<string | null>(null);
  const [iteration, setIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(2); // default fallback
  
  // Combined loading state: local loading OR global context loading OR cross-tab user loading
  // This ensures loading state persists across navigation AND across browser tabs for the same user
  const isAnyLoading = isLoading || cacheLoading || isUserLoading;
  
  // Track previous chatId and message count for scroll logic
  const prevChatIdRef = React.useRef<string | null>(null);
  const prevMsgCountRef = React.useRef<number>(1);
  const inputRef = React.useRef<HTMLTextAreaElement>(null);
  const sidebarRef = React.useRef<HTMLDivElement>(null);
  
  // Simple text persistence
  const setCurrentMessageWithPersistence = (message: string) => {
    setCurrentMessage(message);
    localStorage.setItem('czsu-draft-message', message);
  };

  // NEW: Load threads using pagination instead of loading all at once
  const loadThreadsWithPagination = useCallback(async () => {
    if (!userEmail) {
      console.log('[ChatPage-loadThreads] ⚠ No user email available');
      return;
    }

    // NEW: Prevent concurrent loading for the same user across tabs
    const isAlreadyLoading = checkUserLoadingState(userEmail);
    if (isAlreadyLoading) {
      console.log('[ChatPage-loadThreads] 🔒 Another tab is already loading for user:', userEmail, '- waiting...');
      setLoading(true);
      return;
    }

    console.log('[ChatPage-loadThreads] 🚀 Starting paginated loading...');
    setLoading(true);
    
    // Set loading state to prevent other tabs from starting concurrent loads
    setUserLoadingState(userEmail, true);

    try {
      // Reset pagination and load initial threads
      resetPagination();
      await loadInitialThreads();
      
      console.log('[ChatPage-loadThreads] ✅ Initial threads loaded with pagination');
      
      if (isPageRefresh) {
        console.log('[ChatPage-loadThreads] ✅ F5 refresh completed - using new pagination system');
        resetPageRefresh();
      } else {
        console.log('[ChatPage-loadThreads] ✅ Navigation completed - pagination initialized');
      }
    } catch (error) {
      console.error('[ChatPage-loadThreads] ❌ Error loading threads:', error);
    } finally {
      setLoading(false);
      // Clear loading state to allow other tabs to load if needed
      setUserLoadingState(userEmail, false);
    }
  }, [userEmail, checkUserLoadingState, setLoading, setUserLoadingState, resetPagination, loadInitialThreads, isPageRefresh, resetPageRefresh]);

  // OPTIMIZED: Use cached data only - no more individual API calls
  const loadMessagesFromCheckpoint = async (threadId: string) => {
    if (!threadId || !userEmail) {
      return;
    }

    console.log('[ChatPage-loadMessages] 🔄 Loading messages for thread:', threadId);

    // ALWAYS check cache first - bulk loading should have populated everything
    const hasCachedMessages = hasMessagesForThread(threadId);
    
    if (hasCachedMessages) {
      console.log('[ChatPage-loadMessages] ✅ Using cached messages for thread:', threadId, '(loaded via bulk loading)');
      setActiveThreadId(threadId);
      return;
    }

    // If no cached messages, it means either:
    // 1. Bulk loading failed for this thread (empty thread)
    // 2. This is a new thread created after bulk loading (no messages yet)
    // 3. Cache was cleared (rare)
    console.log('[ChatPage-loadMessages] ⚠ No cached messages found for thread:', threadId);
    
    // Check if we're still loading threads/bulk data
    if (threadsLoading || isLoading) {
      console.log('[ChatPage-loadMessages] 🔄 Still loading bulk data - waiting for completion...');
      // Set the active thread anyway - messages will appear when bulk loading completes
      setActiveThreadId(threadId);
      return;
    }

    // OPTIMIZATION: No more fallback API calls - bulk loading handles everything
    console.log('[ChatPage-loadMessages] 💡 No cached messages for thread:', threadId, '- this thread may be empty or bulk loading failed');
    console.log('[ChatPage-loadMessages] 📊 Bulk loading should have loaded all messages - no individual API calls needed');
    
    // Set the active thread anyway - if it's truly empty, the UI will show "No messages"
    // If bulk loading failed, the user can refresh the page to retry
    setActiveThreadId(threadId);
    
    // REMOVED: No more individual API calls
    // The bulk loading via /chat/all-messages should have loaded all messages
    // If there are no cached messages, it means this thread has no messages or bulk loading failed
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

  // Main effect: Handle user authentication and initialize data
  useEffect(() => {
    if (status === "loading") return; // Wait for session to be determined
    
    if (status === "unauthenticated") {
      console.log('[ChatPage-useEffect] 🚫 User not authenticated, redirecting to login');
      return; // Let the AuthGuard handle the redirect
    }
    
    if (status === "authenticated" && userEmail) {
      console.log('[ChatPage-useEffect] ✅ User authenticated:', userEmail);
      
      // Initialize user in context
      setUserEmail(userEmail);
      
      // Check for user change (different from cached user)
      const existingCache = typeof localStorage !== 'undefined' ? localStorage.getItem('czsu-chat-cache') : null;
      if (existingCache) {
        try {
          const cached = JSON.parse(existingCache);
          if (cached.userEmail && cached.userEmail !== userEmail) {
            console.log('[ChatPage-useEffect] 👤 User changed from', cached.userEmail, 'to', userEmail);
            console.log('[ChatPage-useEffect] 🧹 Clearing cache and resetting pagination for user change');
            
            clearCacheForUserChange(userEmail);
            resetPagination();
            
            console.log('[ChatPage-useEffect] ✅ Previous user data cleared - loading fresh data for current user');
            loadThreadsWithPagination();
            return;
          } else {
            console.log('[ChatPage-useEffect] ✅ Same user as cached data - checking if refresh needed');
          }
        } catch (e) {
          // If there's any issue parsing cache, clear it for safety
          clearCacheForUserChange(userEmail);
          resetPagination();
          loadThreadsWithPagination();
          return;
        }
      }
      
      // Check if we need to refresh from API (F5 or stale data)
      if (isDataStale() || isPageRefresh) {
        console.log('[ChatPage-useEffect] 🔄 Cache is stale or page refresh detected - using pagination');
        resetPagination();
        loadThreadsWithPagination();
      } else {
        console.log('[ChatPage-useEffect] ⚡ Using cached data - no API call needed');
        // Cache is valid and data should already be loaded by ChatCacheContext
        // Threads are considered loaded when we have totalThreadsCount > 0
        
        // If we have cached threads, we need to trigger the UI update
        if (threads.length > 0) {
          console.log('[ChatPage-useEffect] 📤 Found', threads.length, 'cached threads - UI ready');
          
          // Restore active thread from localStorage if not already set
          if (!activeThreadId) {
            const lastActiveThread = localStorage.getItem('czsu-last-active-chat');
            if (lastActiveThread && threads.find(t => t.thread_id === lastActiveThread)) {
              console.log('[ChatPage-useEffect] 🔄 Restoring cached active thread:', lastActiveThread);
              setActiveThreadId(lastActiveThread);
            } else if (threads.length > 0) {
              // Select the most recent thread if no stored thread
              const mostRecentThread = threads[0]; // threads are sorted by latest_timestamp DESC
              console.log('[ChatPage-useEffect] 🔄 No stored thread, selecting most recent:', mostRecentThread.thread_id);
              setActiveThreadId(mostRecentThread.thread_id);
            }
          }
        } else {
          console.log('[ChatPage-useEffect] ⚠ No cached threads found - initializing pagination');
          // Initialize pagination for first-time users
          loadThreadsWithPagination();
        }
      }
    }
  }, [userEmail, status, clearCacheForUserChange, isDataStale, isPageRefresh, threads.length, activeThreadId, setUserEmail, resetPagination, loadThreadsWithPagination, setActiveThreadId]);

  // NEW: Initialize currentMessage from localStorage when user authenticates
  useEffect(() => {
    if (userEmail && status === "authenticated") {
      setCurrentMessage(localStorage.getItem('czsu-draft-message') || '');
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
    if (activeThreadId && totalThreadsCount > 0) {
      console.log('[ChatPage-useEffect] 🔄 Active thread changed, loading messages for:', activeThreadId);
      // Save active thread to localStorage
      localStorage.setItem('czsu-last-active-chat', activeThreadId);
      loadMessagesFromCheckpoint(activeThreadId);
    }
  }, [activeThreadId, totalThreadsCount]);

  // Restore active thread from localStorage after threads are loaded
  useEffect(() => {
    if (totalThreadsCount > 0 && threads.length > 0 && !activeThreadId) {
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
  }, [totalThreadsCount, threads.length, activeThreadId]);

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

  const handlePDFButtonClick = (msgId: string) => {
    setOpenPDFModalForMsgId(msgId);
  };

  const handleClosePDFModal = () => {
    setOpenPDFModalForMsgId(null);
  };

  const handleNewChat = async () => {
    if (!userEmail) {
      return;
    }

    console.log('[ChatPage-newChat] 🔄 Checking for existing New Chat');
    
    // First, check if there's already an existing "New Chat" thread
    const existingNewChat = threads.find(thread => 
      thread.title === 'New Chat' && 
      (!thread.full_prompt || thread.full_prompt.trim() === '')
    );
    
    if (existingNewChat) {
      console.log('[ChatPage-newChat] ✅ Found existing New Chat, navigating to:', existingNewChat.thread_id);
      setActiveThreadId(existingNewChat.thread_id);
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
    localStorage.removeItem('czsu-draft-message'); // Clear saved draft
    
    // Capture state for recovery mechanism
    const messagesBefore = messages.length;
    
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

      // MEMORY PRESSURE HANDLING: Promise.race with timeout monitor for automatic recovery
      console.log('[ChatPage-send] 🚀 Starting API call with memory pressure monitoring...');
      
      // Track timeout ID separately to avoid circular reference
      let timeoutId: NodeJS.Timeout | null = null;
      
      // Create timeout monitor for automatic recovery
      const timeoutMonitor = new Promise<AnalyzeResponse>((resolve, reject) => {
        timeoutId = setTimeout(async () => {
          console.log('[ChatPage-TimeoutMonitor] ⏰ 5 minute timeout reached - checking PostgreSQL for completion');
          
          try {
            // Check if backend completed and saved to PostgreSQL
            const recoverySuccessful = await checkForNewMessagesAfterTimeout(currentThreadId, messagesBefore);
            
            if (recoverySuccessful) {
              console.log('[ChatPage-TimeoutMonitor] ✅ Recovery successful from PostgreSQL - backend completed but HTTP response was stuck');
              
              // Create a synthetic response since we recovered from PostgreSQL
              resolve({
                prompt: messageText,
                result: "Response recovered from database - check messages for actual content",
                queries_and_results: [],
                thread_id: currentThreadId,
                top_selection_codes: [],
                iteration: 0,
                max_iterations: 2,
                sql: null,
                datasetUrl: null,
                run_id: "recovered",
                recovery_mode: true
              } as AnalyzeResponse);
            } else {
              console.log('[ChatPage-TimeoutMonitor] ⏳ Backend still processing - extending timeout and keeping loading state');
              // Don't reject immediately - extend the timeout and keep trying
              // This prevents the loading visual from disappearing while backend is working
              const extendedRecovery = new Promise<AnalyzeResponse>((extendResolve, extendReject) => {
                setTimeout(async () => {
                  console.log('[ChatPage-TimeoutMonitor] ⏰ Extended timeout (8 minutes total) - final recovery check');
                  try {
                    const finalRecovery = await checkForNewMessagesAfterTimeout(currentThreadId, messagesBefore);
                    if (finalRecovery) {
                      console.log('[ChatPage-TimeoutMonitor] ✅ Final recovery successful');
                      extendResolve({
                        prompt: messageText,
                        result: "Response recovered from database after extended processing",
                        queries_and_results: [],
                        thread_id: currentThreadId,
                        top_selection_codes: [],
                        iteration: 0,
                        max_iterations: 2,
                        sql: null,
                        datasetUrl: null,
                        run_id: "recovered",
                        recovery_mode: true
                      } as AnalyzeResponse);
                    } else {
                      extendReject(new Error('Final timeout: No response after 8 minutes'));
                    }
                  } catch (error) {
                    extendReject(error);
                  }
                }, 180000); // Additional 3 minutes (5 + 3 = 8 minutes total)
              });
              
              const result = await extendedRecovery;
              resolve(result);
            }
          } catch (error) {
            console.log('[ChatPage-TimeoutMonitor] ❌ Error during recovery check:', error);
            reject(error);
          }
        }, 300000); // Increased from 15000 (15 seconds) to 300000 (5 minutes)
      });
      
      // Create API call promise
      const apiCall = authApiFetch<AnalyzeResponse>('/analyze', freshSession.id_token, {
        method: 'POST',
        body: JSON.stringify({
          prompt: messageText,
          thread_id: currentThreadId
        }),
      });
      
      // CRITICAL: Use Promise.race to handle both API response and timeout recovery
      console.log('[ChatPage-send] 🏁 Racing API call against timeout monitor...');
      const data = await Promise.race([apiCall, timeoutMonitor]);
      
      // Clear timeout if API call succeeded
      if (timeoutId) {
        clearTimeout(timeoutId);
        console.log('[ChatPage-send] ✅ API call succeeded, timeout monitor cleared');
      }

      console.log('[ChatPage-send] ✅ Response received with run_id:', data.run_id);

      // Debug logging for PDF chunks
      console.log('[ChatPage-send] 🔍 PDF chunks in response:', data.top_chunks?.length || 0);
      if (data.top_chunks && data.top_chunks.length > 0) {
        console.log('[ChatPage-send] 📄 First chunk preview:', data.top_chunks[0].content?.substring(0, 100) + '...');
      }

      // Update loading message with response
      const responseMessage: ChatMessage = {
        id: loadingMessageId,
        threadId: currentThreadId,
        user: 'assistant',
        content: data.result,
        isUser: false,
        createdAt: Date.now(),
        isLoading: false,
        queriesAndResults: data.queries_and_results || [],
        meta: {
          datasetsUsed: data.top_selection_codes || [],
          sqlQuery: data.sql || null,
          iteration: data.iteration || 0,
          maxIterations: data.max_iterations || 2,
          datasetUrl: data.datasetUrl || null,
          runId: data.run_id,
          topChunks: data.top_chunks || []
        }
      };

      // Debug logging for message meta
      console.log('[ChatPage-send] 📋 Message meta topChunks:', responseMessage.meta?.topChunks?.length || 0);
      if (responseMessage.meta && responseMessage.meta.topChunks && responseMessage.meta.topChunks.length > 0) {
        console.log('[ChatPage-send] 📄 Message meta first chunk:', responseMessage.meta.topChunks[0].content?.substring(0, 100) + '...');
      }

      // CRITICAL DEBUG: Log datasetsUsed before update
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - datasetsUsed:', responseMessage.meta?.datasetsUsed);
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - full meta:', JSON.stringify(responseMessage.meta, null, 2));
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - loadingMessageId:', loadingMessageId);
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - currentThreadId:', currentThreadId);

      updateMessage(currentThreadId, loadingMessageId, responseMessage);
      
      // CRITICAL DEBUG: Add a small delay and check the message state
      setTimeout(() => {
        const currentMessages = messages;
        const updatedMessage = currentMessages.find(msg => msg.id === loadingMessageId);
        console.log('[ChatPage-send] 🔍 AFTER updateMessage - found updated message:', !!updatedMessage);
        if (updatedMessage) {
          console.log('[ChatPage-send] 🔍 AFTER updateMessage - datasetsUsed:', updatedMessage.meta?.datasetsUsed);
          console.log('[ChatPage-send] 🔍 AFTER updateMessage - full meta:', JSON.stringify(updatedMessage.meta, null, 2));
        } else {
          console.log('[ChatPage-send] ⚠️ AFTER updateMessage - message not found with ID:', loadingMessageId);
          console.log('[ChatPage-send] ⚠️ AFTER updateMessage - current message IDs:', currentMessages.map(m => m.id));
        }
      }, 100);

      setIsLoading(false);
      
      // IMPORTANT: Mark this response as successfully processed to prevent recovery interference
      console.log('[ChatPage-send] ✅ API response processed successfully - recovery mechanisms should not interfere');
      
    } catch (error) {
      console.error('[ChatPage-send] ❌ Error sending message:', error);
      console.error('[ChatPage-send] ❌ Error details:', {
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : 'No stack trace',
        type: typeof error,
        error: error
      });
      
      // MEMORY PRESSURE RECOVERY: Check if response was saved to PostgreSQL despite error
      console.log('[ChatPage-Recovery] 🔄 Attempting response recovery due to error...');
      
      const recoverySuccessful = await checkForNewMessagesAfterTimeout(currentThreadId, messagesBefore);
      
      if (recoverySuccessful) {
        console.log('[ChatPage-Recovery] 🎉 Response recovered from PostgreSQL!');
        
        // Remove the loading message since we recovered the real response
        console.log('[ChatPage-Recovery] ✅ Response recovery completed successfully');
        
      } else {
        // Recovery failed - show error message
        const errorMessage: ChatMessage = {
          id: loadingMessageId,
          threadId: currentThreadId,
          user: 'assistant',
          content: 'I apologize, but I encountered an issue while processing your request. This might be due to high server load. Please try again, or refresh the page to see if your response was saved.',
          isUser: false,
          createdAt: Date.now(),
          isLoading: false,
          isError: true,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
        
        updateMessage(currentThreadId, loadingMessageId, errorMessage);
        console.log('[ChatPage-Recovery] ❌ Recovery failed - showing error message');
      }
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

  // Focus input when active thread changes AND we have loaded threads
  useEffect(() => {
    if (activeThreadId && totalThreadsCount > 0) {
      console.log('[ChatPage-focusInput] 🎯 Active thread changed, focusing input');
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [activeThreadId, totalThreadsCount]);

  // Set default active thread when threads are loaded
  useEffect(() => {
    if (totalThreadsCount > 0 && threads.length > 0 && !activeThreadId) {
      console.log('[ChatPage-setDefaultThread] 🎯 Setting default active thread to first loaded thread');
      setActiveThreadId(threads[0].thread_id);
    }
  }, [totalThreadsCount, threads.length, activeThreadId]);

  // NEW: Response recovery mechanism for memory pressure scenarios
  const checkForNewMessagesAfterTimeout = async (threadId: string, beforeMessageCount: number) => {
    console.log('[ChatPage-Recovery] 🔄 Checking for new messages after timeout/error for thread:', threadId);
    console.log('[ChatPage-Recovery] 📊 Messages before request:', beforeMessageCount);
    
    try {
      // Small delay to allow PostgreSQL writes to complete
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Load fresh messages from PostgreSQL via our bulk endpoint
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        console.log('[ChatPage-Recovery] ⚠ No session available for recovery check');
        return false;
      }
      
      // Get fresh data from API to check for new messages
      const response = await authApiFetch<{
        messages: { [threadId: string]: any[] };
        runIds: { [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] };
        sentiments: { [threadId: string]: { [runId: string]: boolean } };
      }>('/chat/all-messages', freshSession.id_token);
      
      const freshMessages = response.messages[threadId] || [];
      console.log('[ChatPage-Recovery] 📄 Fresh messages from PostgreSQL:', freshMessages.length);
      
      if (freshMessages.length > beforeMessageCount) {
        console.log('[ChatPage-Recovery] 🎉 RECOVERY SUCCESS: Found new messages! Updating cache...');
        
        // Update cache with fresh data
        setMessages(threadId, freshMessages);
        
        // Update run-ids and sentiments if available
        if (response.runIds[threadId]) {
          console.log('[ChatPage-Recovery] 📝 Updating cached run-ids and sentiments');
        }
        
        console.log('[ChatPage-Recovery] ✅ Chat recovered successfully - new messages are now visible');
        return true;
      } else {
        console.log('[ChatPage-Recovery] ⚠ No new messages found - request may have truly failed');
        return false;
      }
      
    } catch (error) {
      console.error('[ChatPage-Recovery] ❌ Error during message recovery:', error);
      return false;
    }
  };

  // NEW: Periodic health check to detect and recover missing responses
  useEffect(() => {
    if (!userEmail || !activeThreadId) return;
    
    // Only run health check if we have loading messages (indicating potential issues)
    const hasLoadingMessages = messages.some(msg => msg.isLoading);
    if (!hasLoadingMessages) return;
    
    console.log('[ChatPage-HealthCheck] 🔍 Detected loading messages - starting health check');
    
    const healthCheckInterval = setInterval(async () => {
      try {
        // Check if any message has been loading for more than 6 minutes (allowing backend to complete normally)
        const stuckMessages = messages.filter(msg => {
          if (!msg.isLoading || !msg.startedAt) return false;
          const timeElapsed = Date.now() - msg.startedAt;
          return timeElapsed > 360000; // Increased from 120000 (2 minutes) to 360000 (6 minutes)
        });
        
        if (stuckMessages.length > 0) {
          console.log('[ChatPage-HealthCheck] 🚨 Detected stuck loading messages after 6 minutes:', stuckMessages.length);
          console.log('[ChatPage-HealthCheck] 🔄 Attempting automatic recovery...');
          
          const beforeMessageCount = messages.filter(msg => !msg.isLoading).length;
          const recoverySuccessful = await checkForNewMessagesAfterTimeout(activeThreadId, beforeMessageCount);
          
          if (recoverySuccessful) {
            console.log('[ChatPage-HealthCheck] 🎉 Automatic recovery successful!');
          } else {
            console.log('[ChatPage-HealthCheck] ⚠ Automatic recovery failed - response may still be processing');
          }
        }
      } catch (error) {
        console.error('[ChatPage-HealthCheck] ❌ Health check error:', error);
      }
    }, 60000); // Increased from 30000 (30 seconds) to 60000 (60 seconds) to reduce interference
    
    // Cleanup interval after 10 minutes (analysis timeout)
    const cleanupTimer = setTimeout(() => {
      clearInterval(healthCheckInterval);
      console.log('[ChatPage-HealthCheck] ⏰ Health check timeout - stopping automatic recovery');
    }, 600000); // 10 minutes
    
    return () => {
      clearInterval(healthCheckInterval);
      clearTimeout(cleanupTimer);
    };
  }, [activeThreadId, messages, userEmail, checkForNewMessagesAfterTimeout]);

  // NEW: Aggressive refresh mechanism for active analysis
  useEffect(() => {
    if (!userEmail || !activeThreadId) return;
    
    // Check if we have loading messages in the current thread
    const hasLoadingMessages = messages.some(msg => msg.isLoading);
    if (!hasLoadingMessages) return;
    
    console.log('[ChatPage-AggressiveRefresh] 🚀 Starting smart refresh for active analysis');
    
    // Start with less aggressive checking, then increase frequency after backend should be done
    let checkCount = 0;
    
    const aggressiveRefreshInterval = setInterval(async () => {
      try {
        checkCount++;
        
        // For first 4 minutes, check every 30 seconds (less intrusive)
        // After 4 minutes, check every 10 seconds (backend should be near completion)
        const shouldSkip = checkCount < 8 && (checkCount % 6 !== 0); // Only check every 6th time for first 8 intervals (4 minutes)
        
        if (shouldSkip) {
          console.log('[ChatPage-AggressiveRefresh] ⏳ Waiting for backend processing time (check', checkCount, '- skipping)');
          return;
        }
        
        console.log('[ChatPage-AggressiveRefresh] 🔄 Checking for completed analysis (check', checkCount, ')...');
        
        const beforeMessageCount = messages.filter(msg => !msg.isLoading).length;
        const recoverySuccessful = await checkForNewMessagesAfterTimeout(activeThreadId, beforeMessageCount);
        
        if (recoverySuccessful) {
          console.log('[ChatPage-AggressiveRefresh] 🎉 Found completed analysis!');
          // The recovery function will clear the interval by updating messages
        } else {
          console.log('[ChatPage-AggressiveRefresh] ⏳ Analysis still in progress...');
        }
      } catch (error) {
        console.error('[ChatPage-AggressiveRefresh] ❌ Aggressive refresh error:', error);
      }
    }, 5000); // Keep 5-second base interval but skip checks intelligently
    
    // Cleanup interval after 15 minutes (extended analysis timeout)
    const cleanupTimer = setTimeout(() => {
      clearInterval(aggressiveRefreshInterval);
      console.log('[ChatPage-AggressiveRefresh] ⏰ Aggressive refresh timeout - stopping');
    }, 900000); // 15 minutes
    
    return () => {
      clearInterval(aggressiveRefreshInterval);
      clearTimeout(cleanupTimer);
    };
  }, [activeThreadId, messages, userEmail, checkForNewMessagesAfterTimeout]);

  // NEW: Manual refresh function for user-triggered refresh
  const handleManualRefresh = async () => {
    if (!activeThreadId || !userEmail) return;
    
    console.log('[ChatPage-ManualRefresh] 🔄 User triggered manual refresh for thread:', activeThreadId);
    
    try {
      const beforeMessageCount = messages.filter(msg => !msg.isLoading).length;
      const recoverySuccessful = await checkForNewMessagesAfterTimeout(activeThreadId, beforeMessageCount);
      
      if (recoverySuccessful) {
        console.log('[ChatPage-ManualRefresh] 🎉 Manual refresh found new messages!');
      } else {
        console.log('[ChatPage-ManualRefresh] ⚠ No new messages found - analysis may still be in progress');
      }
    } catch (error) {
      console.error('[ChatPage-ManualRefresh] ❌ Manual refresh error:', error);
    }
  };

  // UI
  return (
    <div className="unified-white-block-system">
      {/* Sidebar with its own scroll */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
          <span className="font-bold text-lg text-blue-700">Chats</span>
          <button
            className="px-3 py-1.5 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleNewChat}
            title="New chat"
            disabled={isAnyLoading || !userEmail}
          >
            + New Chat
          </button>
        </div>
        
        {/* Sidebar Chat List with Scroll */}
        <div ref={sidebarRef} className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-1 chat-scrollbar">
          {(threadsLoading && threads.length === 0) ? (
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
            <>
              {threads.map(s => (
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
              ))}
              
              {/* Infinite Scroll Loading Indicator */}
              {threadsHasMore && (
                <div ref={threadsObserverRef} className="w-full">
                  <LoadingSpinner 
                    size="sm" 
                    text="Loading more chats..." 
                    className="py-4"
                  />
                </div>
              )}
              
              {/* Loading indicator for additional pages */}
              {threadsLoading && threads.length > 0 && (
                <LoadingSpinner 
                  size="sm" 
                  text="Loading more chats..." 
                  className="py-2"
                />
              )}
              
              {/* End of list indicator */}
              {!threadsHasMore && threads.length > 10 && (
                <div className="text-center py-4">
                  <div className="text-xs text-gray-400">
                    All {totalThreadsCount} chats loaded
                  </div>
                </div>
              )}
            </>
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
            onPDFClick={handlePDFButtonClick}
            openPDFModalForMsgId={openPDFModalForMsgId}
            onClosePDFModal={handleClosePDFModal}
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
              onChange={e => setCurrentMessageWithPersistence(e.target.value)}
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