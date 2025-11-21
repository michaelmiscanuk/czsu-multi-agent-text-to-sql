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
  
  // Track if we've already attempted to load threads to prevent infinite loops when user has 0 threads
  const [hasAttemptedThreadLoad, setHasAttemptedThreadLoad] = useState(false);
  
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
  
  // State for initial follow-up prompts (for empty chats)
  const [initialFollowupPrompts, setInitialFollowupPrompts] = useState<string[]>([]);
  const [isLoadingInitialPrompts, setIsLoadingInitialPrompts] = useState(false);
  
  // State for tracking current execution for cancellation
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  
  // Combined loading state: local loading OR global context loading OR cross-tab user loading
  // This ensures loading state persists across navigation AND across browser tabs for the same user
  const isAnyLoading = isLoading || cacheLoading || isUserLoading;
  
  // UI blocking loading state: excludes cacheLoading to allow NEW CHAT and input during background refresh
  const isUIBlocking = isLoading || isUserLoading;
  
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
    
    // Mark that we've attempted to load threads (prevents infinite loops for users with 0 threads)
    setHasAttemptedThreadLoad(true);

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
  }, [userEmail, checkUserLoadingState, setLoading, setUserLoadingState, resetPagination, loadInitialThreads, isPageRefresh, resetPageRefresh, setHasAttemptedThreadLoad]);

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
    // The bulk loading via /chat/all-messages-for-all-threads should have loaded all messages
    // If there are no cached messages, it means this thread has no messages or bulk loading failed
  };

  const deleteThreadFromPostgreSQL = async (threadId: string) => {
    console.log('[ChatPage-deleteThread] 🔄 Starting delete process for thread:', threadId);
    
    if (!threadId || !userEmail) {
      console.error('[ChatPage-deleteThread] ❌ Missing threadId or userEmail:', { threadId, userEmail });
      return false;
    }
    
    try {
      console.log('[ChatPage-deleteThread] 🔑 Getting fresh session for authentication...');
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        console.error('[ChatPage-deleteThread] ❌ No authentication token available');
        throw new Error('No authentication token available');
      }

      console.log('[ChatPage-deleteThread] 🚀 Making DELETE request to /chat/' + threadId);
      const response = await authApiFetch(`/chat/${threadId}`, freshSession.id_token, {
        method: 'DELETE',
      });

      console.log('[ChatPage-deleteThread] ✅ DELETE request successful:', response);
      console.log('[ChatPage-deleteThread] 📋 Response data:', response);
      
      // Update cache through context
      console.log('[ChatPage-deleteThread] 🔄 Removing thread from cache...');
      removeThread(threadId);
      
      // If this was the active thread, clear it
      if (activeThreadId === threadId) {
        console.log('[ChatPage-deleteThread] 🔄 Clearing active thread (was deleted)');
        setActiveThreadId(null);
      }
      
      console.log('[ChatPage-deleteThread] ✅ Thread deleted successfully');
      return true;
    } catch (error) {
      console.error('[ChatPage-deleteThread] ❌ Error deleting thread:', error);
      console.error('[ChatPage-deleteThread] ❌ Error details:', {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      });
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
            setHasAttemptedThreadLoad(false); // Reset flag for new user
            
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
          setHasAttemptedThreadLoad(false); // Reset flag when clearing cache
          loadThreadsWithPagination();
          return;
        }
      }
      
      // Check if we need to refresh from API (F5 or stale data)
      if (isDataStale() || isPageRefresh) {
        console.log('[ChatPage-useEffect] 🔄 Cache is stale or page refresh detected - using pagination');
        resetPagination();
        setHasAttemptedThreadLoad(false); // Reset flag for fresh load
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
          console.log('[ChatPage-useEffect] ⚠ No cached threads found - checking if we should initialize pagination');
          // Only initialize pagination for first-time users if we haven't already attempted to load
          if (!hasAttemptedThreadLoad) {
            console.log('[ChatPage-useEffect] 🔄 First attempt - initializing pagination for user with no threads');
            loadThreadsWithPagination();
          } else {
            console.log('[ChatPage-useEffect] ✅ Already attempted to load threads - user has 0 threads, no need to retry');
          }
        }
      }
    }
  }, [userEmail, status, clearCacheForUserChange, isDataStale, isPageRefresh, threads.length, activeThreadId, setUserEmail, resetPagination, loadThreadsWithPagination, setActiveThreadId, hasAttemptedThreadLoad]);

  // Reusable function to fetch initial prompts from backend
  const fetchInitialPrompts = useCallback(async () => {
    if (!userEmail || isLoadingInitialPrompts) {
      return;
    }
    
    console.log('[ChatPage-initialPrompts] 🔄 Fetching initial prompts from backend');
    setIsLoadingInitialPrompts(true);
    
    try {
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        console.log('[ChatPage-initialPrompts] ⚠ No session available for fetching prompts');
        return;
      }
      
      console.log('[ChatPage-initialPrompts] 🌐 Making API call to /initial-followup-prompts');
      const prompts = await authApiFetch<string[]>('/initial-followup-prompts', freshSession.id_token);
      console.log('[ChatPage-initialPrompts] 📡 API response received:', prompts);
      
      if (prompts && prompts.length > 0) {
        console.log('[ChatPage-initialPrompts] ✅ Received', prompts.length, 'initial prompts from backend');
        console.log('[ChatPage-initialPrompts] 📝 Prompts:', prompts);
        setInitialFollowupPrompts(prompts);
      } else {
        console.log('[ChatPage-initialPrompts] ⚠ No prompts returned from backend (empty array)');
        setInitialFollowupPrompts([]);
      }
    } catch (error) {
      console.error('[ChatPage-initialPrompts] ❌ Error fetching initial prompts:', error);
      console.error('[ChatPage-initialPrompts] ❌ Error details:', {
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined
      });
      // Don't set prompts on error - let UI show empty state
      setInitialFollowupPrompts([]);
    } finally {
      setIsLoadingInitialPrompts(false);
    }
  }, [userEmail, isLoadingInitialPrompts]);

  // NEW: Initialize currentMessage from localStorage when user authenticates
  useEffect(() => {
    if (userEmail && status === "authenticated") {
      setCurrentMessage(localStorage.getItem('czsu-draft-message') || '');
    }
  }, [userEmail, status]);

  // NEW: Fetch initial follow-up prompts from backend when user authenticates
  useEffect(() => {
    if (userEmail && status === "authenticated" && initialFollowupPrompts.length === 0) {
      fetchInitialPrompts();
    }
  }, [userEmail, status, initialFollowupPrompts.length, fetchInitialPrompts]);

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
    
    // CRITICAL FIX: Clear old prompts immediately to prevent flash of stale prompts
    setInitialFollowupPrompts([]);
    setIsLoadingInitialPrompts(true);
    
    // First, check if there's already an existing "New Chat" thread
    const existingNewChat = threads.find(thread => 
      thread.title === 'New Chat' && 
      (!thread.full_prompt || thread.full_prompt.trim() === '')
    );
    
    if (existingNewChat) {
      console.log('[ChatPage-newChat] ✅ Found existing New Chat, navigating to:', existingNewChat.thread_id);
      setActiveThreadId(existingNewChat.thread_id);
      // Refresh prompts for existing new chat
      await fetchInitialPrompts();
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
    
    // Fetch fresh prompts for the new chat
    await fetchInitialPrompts();
    
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
    
    // Remove confirmation dialog - just delete directly
    const success = await deleteThreadFromPostgreSQL(threadId);
    if (success) {
      console.log('[ChatPage-delete] ✅ Thread deleted successfully');
    } else {
      console.error('[ChatPage-delete] ❌ Failed to delete thread');
      alert('Failed to delete thread. Please try again.');
    }
  };

  const handleDeleteWithEventHandling = async (e: React.MouseEvent, threadId: string) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('[ChatPage-delete] 🔄 Delete button clicked for thread:', threadId);
    console.log('[ChatPage-delete] 📋 Event target:', e.target);
    console.log('[ChatPage-delete] 📋 Current target:', e.currentTarget);
    
    try {
      const success = await deleteThreadFromPostgreSQL(threadId);
      if (success) {
        console.log('[ChatPage-delete] ✅ Thread deleted successfully');
      } else {
        console.error('[ChatPage-delete] ❌ Failed to delete thread');
        alert('Failed to delete thread. Please try again.');
      }
    } catch (error) {
      console.error('[ChatPage-delete] ❌ Error in delete handler:', error);
      alert('Failed to delete thread. Please try again.');
    }
  };

  const handleFollowupPromptClick = async (promptText: string) => {
    console.log('[ChatPage-followup] 🎯 Follow-up prompt clicked:', promptText);
    
    // Directly execute the send logic without using the form handler
    // This avoids the issues with setting state and synthetic events
    await executeSendMessage(promptText);
  };

  const executeSendMessage = async (messageText: string) => {
    console.log('🔍 print__analysis_tracing_debug: 01 - DIRECT SEND: Direct message send triggered');
    
    if (!messageText.trim() || isUIBlocking || !userEmail) return;
    
    console.log('🔍 print__analysis_tracing_debug: 02 - VALIDATION PASSED: Message validation passed');
    
    // CRITICAL: Block if user is already loading in ANY tab before doing anything else
    const existingLoadingState = checkUserLoadingState(userEmail);
    if (existingLoadingState) {
      console.log('[ChatPage-send] 🚫 BLOCKED: User', userEmail, 'is already processing a request in another tab');
      console.log('🔍 print__analysis_tracing_debug: 03 - USER LOADING BLOCKED: User already processing request in another tab');
      return; // Exit immediately - don't allow concurrent requests
    }

    console.log('[ChatPage-send] ✅ No existing loading state found, proceeding with request for user:', userEmail);
    console.log('🔍 print__analysis_tracing_debug: 04 - LOADING STATE CLEAR: No existing loading state, proceeding');
    
    // Clear the input field and draft
    setCurrentMessage("");
    localStorage.removeItem('czsu-draft-message');
    
    console.log('🔍 print__analysis_tracing_debug: 05 - MESSAGE PREPARED: Message text prepared and input cleared');
    
    // Capture state for recovery mechanism - count messages with final answers
    const messagesBefore = messages.filter(msg => msg.final_answer && !msg.isLoading).length;
    
    // Set loading state in BOTH local and context to ensure persistence across navigation
    setIsLoading(true);
    setLoading(true); // Context loading state - persists across navigation
    
    console.log('🔍 print__analysis_tracing_debug: 06 - LOADING STATES SET: Both local and context loading states set');
    
    // NEW: Set cross-tab loading state tied to user email IMMEDIATELY
    setUserLoadingState(userEmail, true);
    console.log('🔍 print__analysis_tracing_debug: 07 - CROSS-TAB LOADING: Cross-tab loading state set for user');
    
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
      console.log('🔍 print__analysis_tracing_debug: 08 - NEW THREAD CREATED: New thread created with ID:', currentThreadId);
    } else {
      console.log('🔍 print__analysis_tracing_debug: 09 - EXISTING THREAD: Using existing thread ID:', currentThreadId);
      // Check if existing thread needs title update
      const currentThread = threads.find(t => t.thread_id === currentThreadId);
      const currentMessages = messages.filter(msg => msg.user === userEmail); // Count only user messages
      
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
        console.log('🔍 print__analysis_tracing_debug: 10 - THREAD TITLE UPDATE: Updated existing thread title');
      }
    }
    
    // Add user message to cache with both prompt and placeholder for final_answer
    const userMessage: ChatMessage = {
      id: uuidv4(),
      threadId: currentThreadId,
      user: userEmail,
      createdAt: Date.now(),
      prompt: messageText,
      final_answer: undefined, // Will be filled with assistant response
      followup_prompts: undefined, // Will be filled with follow-up suggestions
      queries_and_results: [],
      datasets_used: [],
      sql_query: undefined,
      top_chunks: [],
      isLoading: true, // Set to loading initially
      startedAt: Date.now(), // Track when the request started
      isError: false,
      error: undefined
    };

    addMessage(currentThreadId, userMessage);
    console.log('🔍 print__analysis_tracing_debug: 11 - USER MESSAGE ADDED: User message added to cache');
    
    // Store the message ID for later update
    const messageId = userMessage.id;
    
    // Generate run_id on frontend so we can track it immediately for cancellation
    const generatedRunId = uuidv4();
    setCurrentRunId(generatedRunId);
    console.log('[ChatPage-send] 🔍 Generated run_id for tracking:', generatedRunId);
    
    try {
      // Get fresh session for authentication
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }
      console.log('🔍 print__analysis_tracing_debug: 13 - AUTHENTICATION: Fresh session obtained');

      // Create API call promise with proper timeout
      const apiCall = authApiFetch<AnalyzeResponse>('/analyze', freshSession.id_token, {
        method: 'POST',
        body: JSON.stringify({
          prompt: messageText,
          thread_id: currentThreadId,
          run_id: generatedRunId  // Pass the generated run_id to backend
        }),
      });
      
      // Use Promise.race with a simple timeout
      const data = await Promise.race([
        apiCall,
        new Promise<AnalyzeResponse>((_, reject) => {
          setTimeout(() => {
            reject(new Error('API call timeout after 8 minutes'));
          }, 480000); // 8 minutes to match backend timeout
        })
      ]);

      console.log('[ChatPage-send] ✅ Response received with run_id:', data.run_id);
      console.log('[ChatPage-send] 🔍 DEBUG - Full API response:', data);
      console.log('[ChatPage-send] 🔍 DEBUG - followup_prompts in response:', data.followup_prompts);

      // Verify the run_id matches what we sent
      if (data.run_id !== generatedRunId) {
        console.warn('[ChatPage-send] ⚠️ Run ID mismatch! Sent:', generatedRunId, 'Received:', data.run_id);
      }

      // Update loading message with response
      const responseMessage: ChatMessage = {
        id: messageId,
        threadId: currentThreadId,
        user: userEmail, // Use the actual user email
        createdAt: userMessage.createdAt, // Keep original timestamp
        prompt: messageText, // Keep the original prompt
        final_answer: data.result,
        followup_prompts: data.followup_prompts || undefined, // Add follow-up suggestions from API
        queries_and_results: data.queries_and_results || [],
        datasets_used: data.datasets_used || data.top_selection_codes || [], // Fix: use datasets_used first, fallback to top_selection_codes
        sql_query: data.sql || undefined,
        top_chunks: data.top_chunks || [],
        isLoading: false, // CRITICAL: Explicitly set to false
        isError: false,
        error: undefined,
        startedAt: undefined, // Clear the started timestamp
        run_id: data.run_id // Store the run_id for feedback functionality
      };

      // CRITICAL DEBUG: Log datasetsUsed before update
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - datasetsUsed:', responseMessage.datasets_used);
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - full responseMessage:', JSON.stringify(responseMessage, null, 2));
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - messageId:', messageId);
      console.log('[ChatPage-send] 🔍 BEFORE updateMessage - currentThreadId:', currentThreadId);

      // CRITICAL FIX: Always sync with backend after successful API call
      // This is the key fix - the backend saves the data correctly, but frontend cache gets out of sync
      console.log('[ChatPage-send] 💾 CRITICAL FIX: Syncing with backend after successful API response');
      try {
        const freshSession = await getSession();
        if (freshSession?.id_token) {
          const response = await authApiFetch<{
            messages: any[];
            runIds: { run_id: string; prompt: string; timestamp: string }[];
            sentiments: { [runId: string]: boolean };
          }>(`/chat/all-messages-for-one-thread/${currentThreadId}`, freshSession.id_token);
          
          const freshMessages = response.messages || [];
          
          // Replace frontend cache with authoritative backend data
          setMessages(currentThreadId, freshMessages);
        }
      } catch (syncError) {
        console.error('[ChatPage-send] ❌ Backend sync failed:', syncError);
        // Fallback to optimistic update
        console.log('[ChatPage-send] 🔄 Falling back to optimistic frontend update');
        
        // Check if the message exists before updating
        const existingMessage = messages.find(msg => msg.id === messageId);
        console.log('[ChatPage-send] 🔍 FALLBACK - existing message found:', !!existingMessage);
        
        if (existingMessage) {
          updateMessage(currentThreadId, messageId, responseMessage);
        } else {
          // Message lost - add the complete conversation
          console.log('[ChatPage-send] 🏗️ FALLBACK - Adding complete conversation');
          
          // Add user message if missing
          const userMessageExists = messages.find(msg => msg.prompt === data.prompt && msg.user === userEmail);
          if (!userMessageExists) {
            const userMessageRecovery: ChatMessage = {
              id: uuidv4(),
              threadId: currentThreadId,
              user: userEmail,
              createdAt: Date.now() - 1000,
              prompt: data.prompt,
              final_answer: undefined,
              followup_prompts: undefined,
              queries_and_results: [],
              datasets_used: [],
              sql_query: undefined,
              top_chunks: [],
              isLoading: false,
              isError: false,
              error: undefined
            };
            addMessage(currentThreadId, userMessageRecovery);
          }
          
          // Add response message
          addMessage(currentThreadId, responseMessage);
        }
      }

      // Force re-render by updating the loading state
      setIsLoading(false);
      
      // CRITICAL FIX: Clear loading state for any user 
      setUserLoadingState(userEmail, false);
      
      // Clear the run_id after successful completion
      setCurrentRunId(null);
      
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
          id: messageId,
          threadId: currentThreadId,
          user: userEmail,
          createdAt: userMessage.createdAt,
          prompt: messageText,
          final_answer: 'I apologize, but I encountered an issue while processing your request. This might be due to high server load. Please try again, or refresh the page to see if your response was saved.',
          followup_prompts: undefined,
          queries_and_results: [],
          datasets_used: [],
          sql_query: undefined,
          top_chunks: [],
          isLoading: false,
          isError: true,
          error: error instanceof Error ? error.message : 'Unknown error'
        };
        
        updateMessage(currentThreadId, messageId, errorMessage);
        console.log('[ChatPage-Recovery] ❌ Recovery failed - showing error message');
      }
    } finally {
      setIsLoading(false);
      setLoading(false); // Clear context loading state as well
      
      // NEW: Clear cross-tab loading state tied to user email
      setUserLoadingState(userEmail, false);
      
      // Clear run_id on completion or error
      setCurrentRunId(null);
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    
    console.log('🔍 print__analysis_tracing_debug: 01 - FORM SUBMIT: Form submission triggered');
    
    if (!currentMessage.trim() || isUIBlocking || !userEmail) return;
    
    console.log('🔍 print__analysis_tracing_debug: 02 - VALIDATION PASSED: Message validation passed');
    
    // Execute the message send with the current message text
    const messageText = currentMessage.trim();
    await executeSendMessage(messageText);
  };

  // Add keyboard handler for SHIFT+ENTER vs ENTER
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Only submit if there's content and not loading
      if (currentMessage.trim() && !isUIBlocking) {
        const syntheticEvent = new Event('submit') as any;
        syntheticEvent.preventDefault = () => {};
        handleSend(syntheticEvent);
      }
    }
    // SHIFT+ENTER will naturally create a new line due to default textarea behavior
  };

  // Handle stopping execution
  const handleStopExecution = async () => {
    console.log('[ChatPage-stop] 🔍 Stop button clicked!');
    console.log('[ChatPage-stop] 🔍 currentRunId:', currentRunId);
    console.log('[ChatPage-stop] 🔍 activeThreadId:', activeThreadId);
    console.log('[ChatPage-stop] 🔍 userEmail:', userEmail);
    
    if (!currentRunId || !activeThreadId || !userEmail) {
      console.log('[ChatPage-stop] ⚠ Missing required data for stop request');
      console.log('[ChatPage-stop] ⚠ currentRunId:', currentRunId);
      console.log('[ChatPage-stop] ⚠ activeThreadId:', activeThreadId);
      console.log('[ChatPage-stop] ⚠ userEmail:', userEmail);
      return;
    }

    console.log('[ChatPage-stop] 🛑 Stopping execution for run_id:', currentRunId);
    
    // OPTIMIZATION: Update UI IMMEDIATELY before API call for instant feedback
    const runIdToCancel = currentRunId;
    const threadIdToCancel = activeThreadId;
    
    // Clear loading states immediately
    setIsLoading(false);
    setLoading(false);
    setUserLoadingState(userEmail, false);
    setCurrentRunId(null);

    // Update the loading message immediately to show it was cancelled
    const loadingMessages = messages.filter(msg => msg.isLoading);
    if (loadingMessages.length > 0) {
      const loadingMessage = loadingMessages[loadingMessages.length - 1];
      updateMessage(threadIdToCancel, loadingMessage.id, {
        ...loadingMessage,
        isLoading: false,
        isError: true,
        error: 'Execution cancelled by user',
        final_answer: 'Analysis was cancelled. Please try again if needed.',
      });
    }

    // Send the stop request in the background (don't await)
    // This allows the UI to update instantly while the cancellation happens asynchronously
    (async () => {
      try {
        const freshSession = await getSession();
        if (!freshSession?.id_token) {
          console.error('[ChatPage-stop] ❌ No authentication token available');
          return;
        }

        // Call the stop endpoint
        await authApiFetch(`/stop-execution`, freshSession.id_token, {
          method: 'POST',
          body: JSON.stringify({
            thread_id: threadIdToCancel,
            run_id: runIdToCancel,
          }),
        });

        console.log('[ChatPage-stop] ✅ Stop request sent successfully');
      } catch (error) {
        console.error('[ChatPage-stop] ❌ Error stopping execution:', error);
        // Note: UI already updated, so we just log the error
      }
    })();
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
      // Use the single-thread endpoint for efficiency
      console.log('[ChatPage-Recovery] 📡 Calling single-thread endpoint for recovery');
      const response = await authApiFetch<{
        messages: any[];
        runIds: { run_id: string; prompt: string; timestamp: string }[];
        sentiments: { [runId: string]: boolean };
      }>(`/chat/all-messages-for-one-thread/${threadId}`, freshSession.id_token);
      
      const freshMessages = response.messages || [];
      console.log('[ChatPage-Recovery] 📄 Fresh messages from PostgreSQL:', freshMessages.length);
      console.log('[ChatPage-Recovery] 🔍 DEBUG - Fresh messages content:', JSON.stringify(freshMessages, null, 2));
      console.log('[ChatPage-Recovery] 🔍 DEBUG - Current messages:', JSON.stringify(messages, null, 2));
      
      // Check if we have new content (final_answer populated) rather than just new message count
      // Since IDs may not match between frontend (UUID) and backend (sequential), 
      // let's check for content changes in a more robust way
      
      const currentCompletedAnswers = messages.filter(msg => msg.final_answer && !msg.isLoading).length;
      const freshCompletedAnswers = freshMessages.filter(msg => msg.final_answer).length;
      
      const hasNewContent = freshCompletedAnswers > currentCompletedAnswers;
      
      console.log('[ChatPage-Recovery] 🔍 Content comparison:', {
        currentCompletedAnswers,
        freshCompletedAnswers,
        hasNewContent,
        currentMessagesTotal: messages.length,
        freshMessagesTotal: freshMessages.length
      });
      
      // Also log first few messages for debugging
      freshMessages.slice(0, 2).forEach((msg, i) => {
        console.log(`[ChatPage-Recovery] 🔍 Fresh message ${i}:`, {
          id: msg.id,
          hasPrompt: !!msg.prompt,
          promptPreview: msg.prompt ? msg.prompt.substring(0, 50) + '...' : 'none',
          hasFinalAnswer: !!msg.final_answer,
          finalAnswerPreview: msg.final_answer ? msg.final_answer.substring(0, 50) + '...' : 'none'
        });
      });
      
      const hasMoreMessages = freshMessages.length > beforeMessageCount;
      
      console.log('[ChatPage-Recovery] 🔍 Recovery check - hasMoreMessages:', hasMoreMessages, 'hasNewContent:', hasNewContent);
      
      if (hasMoreMessages || hasNewContent) {
        console.log('[ChatPage-Recovery] 🎉 RECOVERY SUCCESS: Found new messages or content! Updating cache...');
        
        // Update cache with fresh data
        setMessages(threadId, freshMessages);
        
        // Update run-ids and sentiments if available
        if (response.runIds) {
          console.log('[ChatPage-Recovery] 📝 Updating cached run-ids and sentiments');
          // This part needs a new context function if we want to update runIds and sentiments from here
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

  // UI
  return (
    <div className="unified-white-block-system">
      {/* Sidebar with its own scroll */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
          <span className="font-bold text-lg text-blue-700">Chats</span>
          <div className="flex gap-2">
            <button
              className="px-3 py-1.5 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleNewChat}
              title="New chat"
              disabled={isUIBlocking || !userEmail}
            >
              + New Chat
            </button>
          </div>
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
              {threads.filter((thread, index, arr) => arr.findIndex(t => t.thread_id === thread.thread_id) === index).map(s => (
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
                        style={{fontFamily: 'var(--font-family)'}}
                        onClick={() => setActiveThreadId(s.thread_id)}
                        onDoubleClick={() => { setEditingTitleId(s.thread_id); setNewTitle(s.title || ''); }}
                        title={`${s.full_prompt || s.title || 'New Chat'}${s.full_prompt && s.full_prompt.length === 50 ? '...' : ''}`}
                      >
                        <div className="truncate block leading-tight">{s.title || 'New Chat'}</div>
                      </button>
                      <button
                        className="flex-shrink-0 ml-1 text-gray-400 hover:text-red-500 text-lg font-bold px-2 py-1 rounded transition-colors cursor-pointer"
                        title="Delete chat"
                        onClick={(e) => handleDeleteWithEventHandling(e, s.thread_id)}
                        style={{ pointerEvents: 'auto', userSelect: 'none' }}
                      >
                        ×
                      </button>
                    </div>
                  )}
                </div>
              ))}
              
              {/* Unified Loading Indicator - prevents duplicates */}
              {threadsLoading && threads.length > 0 ? (
                // Show loading indicator for additional pages when already loading
                <LoadingSpinner 
                  size="sm" 
                  text="Loading more chats..." 
                  className="py-4"
                />
              ) : threadsHasMore ? (
                // Show infinite scroll observer when not loading but has more
                <div ref={threadsObserverRef} className="w-full">
                  <LoadingSpinner 
                    size="sm" 
                    text="Loading more chats..." 
                    className="py-4"
                  />
                </div>
              ) : null}
              
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
            onFollowupPromptClick={handleFollowupPromptClick}
            onRerunPrompt={executeSendMessage}
            isLoading={isAnyLoading}
            isAnyLoading={isUIBlocking}
            threads={threads}
            activeThreadId={activeThreadId}
          />
        </div>
        
        {/* Follow-up Prompts Section - Show initial prompts for new chats only */}
        {(() => {
          let followupPrompts: string[] = [];
          
          // Only show initial prompts if there are NO messages (new chat)
          if (messages.length === 0) {
            followupPrompts = initialFollowupPrompts;
            console.log('[ChatPage-followup] Using initial prompts for new chat:', followupPrompts);
          }
          // If there are messages, followup prompts are handled inside MessageArea
          
          console.log('[ChatPage-followup] Final follow-up prompts to display in initial section:', followupPrompts);
          
          // Show loading state while fetching new prompts
          if (messages.length === 0 && isLoadingInitialPrompts) {
            return (
              <div className="bg-white border-t border-gray-100 py-3">
                <div className="max-w-4xl mx-auto px-4">
                  <div className="flex items-center justify-center py-8">
                    <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <span className="ml-3 text-sm text-gray-500">Loading suggestions...</span>
                  </div>
                </div>
              </div>
            );
          }
          
          return followupPrompts.length > 0 ? (
            <div className="bg-white border-t border-gray-100 py-3">
              <div className="max-w-4xl mx-auto px-4">
                <div className="flex flex-col gap-2">
                  {followupPrompts.map((prompt, index) => (
                    <button
                      key={index}
                      onClick={() => handleFollowupPromptClick(prompt)}
                      disabled={isUIBlocking}
                      className="group w-full text-left px-4 py-3 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-xl transition-all duration-200 flex items-center justify-between disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        fontFamily: 'var(--font-family)',
                      }}
                    >
                      <span className="text-sm text-gray-700 group-hover:text-gray-900 flex-1">
                        {prompt}
                      </span>
                      <svg
                        className="w-5 h-5 text-gray-400 group-hover:text-gray-600 flex-shrink-0 ml-3"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 7l5 5m0 0l-5 5m5-5H6"
                        />
                      </svg>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : null;
        })()}
        
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
              disabled={isUIBlocking}
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
            <div className="flex items-center gap-2">
              <button
                type="submit"
                className="px-6 py-3 light-blue-theme rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 mt-1 flex items-center justify-center min-w-[80px]"
                disabled={isAnyLoading || !currentMessage.trim()}
              >
                {isAnyLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <span>Send</span>
                )}
              </button>
              {isAnyLoading && (
                <button
                  type="button"
                  onClick={handleStopExecution}
                  className="px-4 py-3 light-blue-theme rounded-xl font-semibold transition-all duration-200 mt-1 flex items-center justify-center"
                  title="Stop execution"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="1" />
                  </svg>
                </button>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}