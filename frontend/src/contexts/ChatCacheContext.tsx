'use client'

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { ChatThreadMeta, ChatMessage, PaginatedChatThreadsResponse } from '@/types'
import { API_CONFIG, authApiFetch } from '@/lib/api'
import { getSession } from "next-auth/react"

interface CacheData {
  threads: ChatThreadMeta[];
  messages: { [threadId: string]: ChatMessage[] };
  runIds: { [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] };
  sentiments: { [threadId: string]: { [runId: string]: boolean } };
  activeThreadId: string | null;
  lastUpdated: number;
  userEmail: string | null;
}

interface ChatCacheContextType {
  // State
  threads: ChatThreadMeta[];
  messages: ChatMessage[];
  activeThreadId: string | null;
  isLoading: boolean;
  
  // Pagination state
  threadsPage: number;
  threadsHasMore: boolean;
  threadsLoading: boolean;
  totalThreadsCount: number;
  
  // Actions
  setThreads: (threads: ChatThreadMeta[]) => void;
  setMessages: (threadId: string, messages: ChatMessage[]) => void;
  setActiveThreadId: (threadId: string | null) => void;
  addMessage: (threadId: string, message: ChatMessage) => void;
  updateMessage: (threadId: string, messageId: string, updatedMessage: ChatMessage) => void;
  addThread: (thread: ChatThreadMeta) => void;
  removeThread: (threadId: string) => void;
  updateThread: (threadId: string, updates: Partial<ChatThreadMeta>) => void;
  
  // Pagination actions
  loadInitialThreads: () => Promise<void>;
  loadMoreThreads: () => Promise<void>;
  resetPagination: () => void;
  
  // Cache management
  invalidateCache: () => void;
  refreshFromAPI: () => Promise<void>;
  isDataStale: () => boolean;
  
  // Load state
  setLoading: (loading: boolean) => void;
  
  // NEW: Track if this is a page refresh
  isPageRefresh: boolean;
  
  // NEW: Force API refresh on F5
  forceAPIRefresh: () => Promise<void>;
  
  // NEW: Check if messages exist for a specific thread
  hasMessagesForThread: (threadId: string) => boolean;
  
  // NEW: Reset page refresh state
  resetPageRefresh: () => void;
  
  // NEW: Cross-tab loading state management
  isUserLoading: boolean;
  setUserLoadingState: (email: string, loading: boolean) => void;
  checkUserLoadingState: (email: string) => boolean;
  setUserEmail: (email: string | null) => void;
  
  // NEW: Clear cache for user changes (logout/login)
  clearCacheForUserChange: (newUserEmail?: string | null) => void;
  
  // NEW: Bulk load all messages, run-ids, and sentiments at once
  loadAllMessagesFromAPI: () => Promise<void>;
  
  // NEW: Get cached run-ids and sentiments for a thread
  getRunIdsForThread: (threadId: string) => { run_id: string; prompt: string; timestamp: string }[];
  getSentimentsForThread: (threadId: string) => { [runId: string]: boolean };
  
  // NEW: Update cached sentiment
  updateCachedSentiment: (threadId: string, runId: string, sentiment: boolean | null) => void;
}

const ChatCacheContext = createContext<ChatCacheContextType | undefined>(undefined)

// Cache configuration
const CACHE_KEY = 'czsu-chat-cache'
const ACTIVE_THREAD_KEY = 'czsu-last-active-chat'
const CACHE_DURATION = 60 * 60 * 1000 * 48 // 48 hours in milliseconds
const PAGE_REFRESH_FLAG_KEY = 'czsu-page-refresh-flag'
const USER_LOADING_STATE_KEY = 'czsu-user-loading-state' // NEW: Cross-tab loading state
const F5_REFRESH_THROTTLE_KEY = 'czsu-f5-refresh-throttle' // NEW: F5 refresh throttling
const F5_REFRESH_COOLDOWN = 100 // Reduced for faster recovery testing (was 1000)

export function ChatCacheProvider({ children }: { children: React.ReactNode }) {
  // Internal state
  const [threads, setThreadsState] = useState<ChatThreadMeta[]>([])
  const [messages, setMessagesState] = useState<{ [threadId: string]: ChatMessage[] }>({})
  const [runIds, setRunIdsState] = useState<{ [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] }>({})
  const [sentiments, setSentimentsState] = useState<{ [threadId: string]: { [runId: string]: boolean } }>({})
  const [activeThreadId, setActiveThreadIdState] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<number>(0)
  const [userEmail, setUserEmail] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isPageRefresh, setIsPageRefresh] = useState(false)
  
  // NEW: Cross-tab loading state tied to user email
  const [isUserLoading, setIsUserLoading] = useState(false)
  
  // NEW: Pagination state
  const [threadsPage, setThreadsPage] = useState(1)
  const [threadsHasMore, setThreadsHasMore] = useState(true)
  const [threadsLoading, setThreadsLoading] = useState(false)
  const [totalThreadsCount, setTotalThreadsCount] = useState(0)
  
  // Track if this is the initial mount to detect page refresh
  const isInitialMount = useRef(true)
  const hasBeenHydrated = useRef(false)

  // Reset page refresh flag after initial mount to ensure subsequent navigation uses cache
  useEffect(() => {
    // Reset page refresh flag after a short delay to ensure it only applies to the initial load
    if (isPageRefresh) {
      const timer = setTimeout(() => {
        console.log('[ChatCache] ⏰ Resetting page refresh flag after initial load - subsequent navigation will use cache');
        setIsPageRefresh(false);
      }, 2000); // 2 second delay to allow initial API calls to complete

      return () => clearTimeout(timer);
    }
  }, [isPageRefresh]);

  // Page refresh detection logic - fix to distinguish F5 from navigation
  useEffect(() => {
    // Ensure we're on the client side
    if (typeof window === 'undefined') return;
    
    console.log('[ChatCache] 🔄 Component mounted, checking if page refresh...');
    
    const now = Date.now();
    
    // Check F5 refresh throttling first
    const lastF5Refresh = typeof localStorage !== 'undefined' ? localStorage.getItem(F5_REFRESH_THROTTLE_KEY) : null;
    const isF5Throttled = lastF5Refresh && (now - parseInt(lastF5Refresh, 10)) < F5_REFRESH_COOLDOWN;
    
    if (isF5Throttled) {
      const timeLeft = F5_REFRESH_COOLDOWN - (now - parseInt(lastF5Refresh, 10));
      const minutesLeft = Math.ceil(timeLeft / (60 * 1000));
      console.log('[ChatCache] 🚫 F5 refresh throttled - must wait', minutesLeft, 'more minutes');
      console.log('[ChatCache] ⚡ Using cached data instead of API refresh');
      loadFromStorage();
      hasBeenHydrated.current = true;
      return;
    }
    
    // More robust page refresh detection
    const isPageReload = () => {
      // Method 1: Performance navigation API (most reliable)
      if (typeof performance !== 'undefined' && performance.navigation && performance.navigation.type === 1) {
        return true;
      }
      
      // Method 2: Performance getEntriesByType (modern browsers)
      if (typeof performance !== 'undefined' && performance.getEntriesByType) {
        const entries = performance.getEntriesByType('navigation');
        if (entries.length > 0) {
          const navigationEntry = entries[0] as PerformanceNavigationTiming;
          if (navigationEntry.type === 'reload') {
            return true;
          }
        }
      }
      
      return false;
    };

    // Check if localStorage has data (previous session)
    const hasLocalStorageData = typeof localStorage !== 'undefined' && !!localStorage.getItem(CACHE_KEY);
    
    // Final decision logic - be much more conservative about detecting refresh
    const actualPageRefresh = isPageReload();
    
    // Only set page refresh if we're confident it's a real F5/reload AND we have existing data
    // This prevents normal navigation from being treated as refresh
    const finalDecision = actualPageRefresh && hasLocalStorageData;
    
    console.log('[ChatCache] 🔍 Page refresh detection: ', {
      performanceNavType: typeof performance !== 'undefined' ? performance.navigation?.type : 'undefined',
      performanceEntries: typeof performance !== 'undefined' ? (performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming)?.type : 'undefined',
      hasLocalStorageData,
      actualPageRefresh,
      finalDecision,
      isF5Throttled
    });
    
    setIsPageRefresh(finalDecision);
    
    // Set F5 refresh timestamp if this is a real refresh
    if (finalDecision && typeof localStorage !== 'undefined') {
      localStorage.setItem(F5_REFRESH_THROTTLE_KEY, now.toString());
      console.log('[ChatCache] 🕐 F5 refresh timestamp recorded - next refresh allowed in 5 minutes');
    }
    
    // Load initial data
    if (finalDecision) {
      console.log('[ChatCache] 🔄 Real page refresh (F5) detected - will force API sync');
    } else {
      console.log('[ChatCache] 🔄 Navigation detected - loading from localStorage cache');
      loadFromStorage();
    }
    
    // Mark as hydrated after initial setup
    hasBeenHydrated.current = true;
  }, []);

  const loadFromStorage = useCallback(() => {
    // Ensure we're on the client side
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') return;
    
    try {
      const stored = localStorage.getItem(CACHE_KEY)
      const activeThread = localStorage.getItem(ACTIVE_THREAD_KEY)
      
      if (stored) {
        const data: CacheData = JSON.parse(stored)
        console.log('[ChatCache] 📤 Loaded from localStorage:', {
          threads: data.threads?.length || 0,
          totalMessages: Object.keys(data.messages || {}).length,
          activeThread: data.activeThreadId,
          lastUpdated: new Date(data.lastUpdated).toISOString(),
          userEmail: data.userEmail
        })
        
        setThreadsState(data.threads || [])
        setMessagesState(data.messages || {})
        setRunIdsState(data.runIds || {})
        setSentimentsState(data.sentiments || {})
        setUserEmail(data.userEmail)
        
        // Restore active thread from either cache or separate storage
        const threadToActivate = activeThread || data.activeThreadId
        if (threadToActivate) {
          setActiveThreadIdState(threadToActivate)
        }
      }
    } catch (error) {
      console.error('[ChatCache] ❌ Error loading from localStorage:', error)
    }
  }, [])

  const saveToStorage = useCallback((data: Partial<CacheData>) => {
    if (!hasBeenHydrated.current) {
      console.log('[ChatCache] ⏳ Skipping save - not yet hydrated')
      return
    }

    // Ensure we're on the client side
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') return;

    try {
      const existingData = localStorage.getItem(CACHE_KEY)
      const existing: CacheData = existingData ? JSON.parse(existingData) : {
        threads: [],
        messages: {},
        runIds: {},
        sentiments: {},
        activeThreadId: null,
        lastUpdated: 0,
        userEmail: null
      }

      const updated: CacheData = {
        ...existing,
        ...data,
        lastUpdated: Date.now()
      }

      localStorage.setItem(CACHE_KEY, JSON.stringify(updated))
      
      // Also save active thread separately for quick access
      if (data.activeThreadId !== undefined) {
        if (data.activeThreadId) {
          localStorage.setItem(ACTIVE_THREAD_KEY, data.activeThreadId)
        } else {
          localStorage.removeItem(ACTIVE_THREAD_KEY)
        }
      }

      console.log('[ChatCache] 💾 Saved to localStorage:', {
        threads: updated.threads?.length || 0,
        totalMessages: Object.keys(updated.messages || {}).length,
        activeThread: updated.activeThreadId,
        userEmail: updated.userEmail
      })
    } catch (error) {
      console.error('[ChatCache] ❌ Error saving to localStorage:', error)
    }
  }, [hasBeenHydrated])

  // Save data whenever state changes
  useEffect(() => {
    if (hasBeenHydrated.current) {
      saveToStorage({
        threads,
        messages,
        runIds,
        sentiments,
        activeThreadId,
        userEmail
      })
    }
  }, [threads, messages, runIds, sentiments, activeThreadId, userEmail, saveToStorage])

  // Actions
  const setThreads = useCallback((newThreads: ChatThreadMeta[]) => {
    console.log('[ChatCache] 🔄 Setting threads:', newThreads.length)
    setThreadsState(newThreads)
  }, [])

  const setMessages = useCallback((threadId: string, newMessages: ChatMessage[]) => {
    console.log('[ChatCache] 🔄 Setting messages for thread:', threadId, 'count:', newMessages.length)
    setMessagesState(prev => ({
      ...prev,
      [threadId]: newMessages
    }))
  }, [])

  const setActiveThreadId = useCallback((threadId: string | null) => {
    console.log('[ChatCache] 🔄 Setting active thread:', threadId)
    setActiveThreadIdState(threadId)
  }, [])

  const addMessage = useCallback((threadId: string, message: ChatMessage) => {
    console.log('[ChatCache] ➕ Adding message to thread:', threadId)
    setMessagesState(prev => ({
      ...prev,
      [threadId]: [...(prev[threadId] || []), message]
    }))
  }, [])

  const updateMessage = useCallback((threadId: string, messageId: string, updatedMessage: ChatMessage) => {
    console.log('[ChatCache] 📝 Updating message:', messageId, 'in thread:', threadId);
    
    setMessagesState(prev => {
      const currentMessages = prev[threadId] || [];
      
      const messageIndex = currentMessages.findIndex(msg => msg.id === messageId);
      
      if (messageIndex === -1) {
        console.log('[ChatCache] ⚠️ UpdateMessage - WARNING: Message not found with ID:', messageId);
        console.log('[ChatCache] 🔍 UpdateMessage - Available message IDs:', currentMessages.map(m => ({ id: m.id, isLoading: m.isLoading })));
        return prev; // Return unchanged state if message not found
      }

      // Create a completely new array to ensure React detects the change
      const updatedMessages = [...currentMessages];
      updatedMessages[messageIndex] = { ...updatedMessage }; // Create a new object reference
      
      return {
        ...prev,
        [threadId]: updatedMessages
      };
    })
  }, [])

  const addThread = useCallback((thread: ChatThreadMeta) => {
    console.log('[ChatCache] ➕ Adding thread:', thread.thread_id)
    setThreadsState(prev => [thread, ...prev])
  }, [])

  const removeThread = useCallback((threadId: string) => {
    console.log('[ChatCache] ➖ Removing thread:', threadId)
    setThreadsState(prev => prev.filter(t => t.thread_id !== threadId))
    setMessagesState(prev => {
      const newMessages = { ...prev }
      delete newMessages[threadId]
      return newMessages
    })
  }, [])

  const updateThread = useCallback((threadId: string, updates: Partial<ChatThreadMeta>) => {
    console.log('[ChatCache] 📝 Updating thread:', threadId, updates)
    setThreadsState(prev => prev.map(thread => 
      thread.thread_id === threadId ? { ...thread, ...updates } : thread
    ))
  }, [])

  const invalidateCache = useCallback(() => {
    console.log('[ChatCache] 🗑️ Invalidating cache')
    localStorage.removeItem(CACHE_KEY)
    localStorage.removeItem(ACTIVE_THREAD_KEY)
    setThreadsState([])
    setMessagesState({})
    setRunIdsState({})
    setSentimentsState({})
    setActiveThreadIdState(null)
    setUserEmail(null)
  }, [])

  const refreshFromAPI = useCallback(async () => {
    console.log('[ChatCache] 🔄 Manual refresh from API requested')
    // This will be implemented by the chat page
    // For now, just invalidate cache to force reload
    invalidateCache()
  }, [invalidateCache])

  const forceAPIRefresh = useCallback(async () => {
    console.log('[ChatCache] 🔄 Force API refresh (F5) - clearing cache and forcing PostgreSQL sync')
    
    // Ensure we're on the client side
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      // Clear all cache data
      localStorage.removeItem(CACHE_KEY)
      localStorage.removeItem(ACTIVE_THREAD_KEY)
    }
    
    // Reset state
    setThreadsState([])
    setMessagesState({})
    setRunIdsState({})
    setSentimentsState({})
    setActiveThreadIdState(null)
    
    // Mark that we need fresh data from API
    setIsLoading(true)
    
    // IMPORTANT: Reset isPageRefresh to false after clearing cache
    // This ensures subsequent navigation uses cache instead of always hitting API
    setIsPageRefresh(false)
    
    console.log('[ChatCache] ✅ Cache cleared - ready for fresh API data')
  }, [])

  const isDataStale = useCallback(() => {
    // If this is a page refresh, always consider data stale to force API call
    if (isPageRefresh) {
      console.log('[ChatCache] 🔍 Data is stale (page refresh detected)')
      return true
    }
    
    // Ensure we're on the client side
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
      console.log('[ChatCache] 🔍 Data is stale (no localStorage access)')
      return true
    }
    
    try {
      const stored = localStorage.getItem(CACHE_KEY)
      if (!stored) {
        console.log('[ChatCache] 🔍 Data is stale (no cache)')
        return true
      }
      
      const data: CacheData = JSON.parse(stored)
      const age = Date.now() - data.lastUpdated
      const isStale = age > CACHE_DURATION
      
      console.log('[ChatCache] 🔍 Data staleness check:', {
        age: Math.round(age / 1000),
        maxAge: Math.round(CACHE_DURATION / 1000),
        isStale
      })
      
      return isStale
    } catch (error) {
      console.error('[ChatCache] ❌ Error checking staleness:', error)
      return true
    }
  }, [isPageRefresh])

  const setLoading = useCallback((loading: boolean) => {
    setIsLoading(loading)
  }, [])

  // NEW: Function to set user email from components
  const setUserEmailContext = useCallback((email: string | null) => {
    console.log('[ChatCache] 👤 Setting user email in context:', email);
    setUserEmail(email);
  }, []);

  // NEW: Clear cache when user changes (for logout/login scenarios)
  const clearCacheForUserChange = useCallback((newUserEmail: string | null = null) => {
    console.log('[ChatCache] 🔄 User change detected - clearing cache for clean state');
    console.log('[ChatCache] 👤 Previous user:', userEmail, '→ New user:', newUserEmail);
    
    // Ensure we're on the client side
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      // Clear all czsu-related localStorage items
      const keysToRemove = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (key.startsWith('czsu-') || key.startsWith('chat-'))) {
          keysToRemove.push(key);
        }
      }
      
      keysToRemove.forEach(key => {
        localStorage.removeItem(key);
        console.log('[ChatCache] 🧹 Cleared localStorage key for user change:', key);
      });
      
      console.log('[ChatCache] ✅ localStorage cleanup completed for user change');
    }
    
    // Reset all state
    setThreadsState([])
    setMessagesState({})
    setRunIdsState({})
    setSentimentsState({})
    setActiveThreadIdState(null)
    setUserEmail(newUserEmail)
    setIsLoading(false)
    setIsPageRefresh(false)
    setIsUserLoading(false)
    
    console.log('[ChatCache] ✅ Cache state reset for user change');
  }, [userEmail]);

  // Get current messages for active thread
  const currentMessages = activeThreadId ? messages[activeThreadId] || [] : []

  // NEW: Cross-tab loading state management
  const setUserLoadingState = useCallback((email: string, loading: boolean) => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const key = `czsu-user-loading-${email}`;
      if (loading) {
        localStorage.setItem(key, Date.now().toString());
        console.log('[ChatCache] 🔒 Set user loading state for:', email);
      } else {
        localStorage.removeItem(key);
        console.log('[ChatCache] 🔓 Cleared user loading state for:', email);
      }
    }
  }, []);

  const checkUserLoadingState = useCallback((email: string): boolean => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const key = `czsu-user-loading-${email}`;
      const loadingTime = localStorage.getItem(key);
      if (loadingTime) {
        const elapsed = Date.now() - parseInt(loadingTime, 10);
        // Consider loading stale after 30 seconds
        if (elapsed > 30000) {
          localStorage.removeItem(key);
          console.log('[ChatCache] ⏰ User loading state expired for:', email);
          return false;
        }
        console.log('[ChatCache] 🔒 User is already loading:', email, 'elapsed:', elapsed, 'ms');
        return true;
      }
    }
    return false;
  }, []);

  // NEW: Reset page refresh state
  const resetPageRefresh = useCallback(() => {
    console.log('[ChatCache] 🔄 Resetting page refresh state');
    setIsPageRefresh(false);
  }, []);

  // NEW: Bulk load all messages, run-ids, and sentiments at once
  const loadAllMessagesFromAPI = useCallback(async () => {
    console.log('[ChatCache] 🔄 Loading ALL messages from API using bulk endpoint...');
    setIsLoading(true);
    
    try {
      // Import authApiFetch and getSession from the lib/api file
      const { authApiFetch } = await import('@/lib/api');
      const { getSession } = await import('next-auth/react');
      
      // Get fresh session - the authApiFetch will handle token refresh if needed
      console.log('[ChatCache] 🔐 Getting fresh session for bulk loading...');
      const freshSession = await getSession();
      
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available for bulk loading');
      }
      
      // OPTIMIZATION: Load all data at once with better error handling
      console.log('[ChatCache] 📡 Calling /chat/all-messages-for-all-threads endpoint with automatic token refresh...');
      const startTime = Date.now();
      
      const response = await authApiFetch<{
        messages: { [threadId: string]: any[] };
        runIds: { [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] };
        sentiments: { [threadId: string]: { [runId: string]: boolean } };
      }>('/chat/all-messages-for-all-threads', freshSession.id_token);
      
      const loadTime = Date.now() - startTime;
      const totalThreadsWithMessages = Object.keys(response.messages || {}).length;
      const totalMessages = Object.values(response.messages || {}).reduce((sum, msgs) => sum + msgs.length, 0);
      
      console.log('[ChatCache] ✅ Loaded ALL data in', loadTime, 'ms:', {
        messageThreads: totalThreadsWithMessages,
        totalMessages: totalMessages,
        runIdThreads: Object.keys(response.runIds || {}).length,
        sentimentThreads: Object.keys(response.sentiments || {}).length
      });
      
      // Debug: Check if messages have followup_prompts
      console.log('[ChatCache] 🔍 DEBUG - Checking followup_prompts in API response:');
      Object.entries(response.messages || {}).forEach(([threadId, msgs]) => {
        const msgsWithFollowup = msgs.filter(m => m.followup_prompts && m.followup_prompts.length > 0);
        if (msgsWithFollowup.length > 0) {
          console.log(`[ChatCache] 🔍 Thread ${threadId} has ${msgsWithFollowup.length} messages with followup_prompts:`, 
            msgsWithFollowup.map(m => ({ id: m.id, followup_prompts: m.followup_prompts }))
          );
        } else {
          console.log(`[ChatCache] ⚠️ Thread ${threadId} has NO messages with followup_prompts (checked ${msgs.length} messages)`);
        }
      });
      
      // OPTIMIZATION: Only update state if we have meaningful data
      if (totalMessages > 0 || Object.keys(response.runIds || {}).length > 0) {
        console.log('[ChatCache] 💾 Storing bulk data in cache...');
        setMessagesState(response.messages || {});
        setRunIdsState(response.runIds || {});
        setSentimentsState(response.sentiments || {});
        
        // Force save to localStorage immediately after bulk loading
        saveToStorage({
          messages: response.messages || {},
          runIds: response.runIds || {},
          sentiments: response.sentiments || {}
        });
        
        console.log('[ChatCache] ✅ Bulk loading completed successfully with automatic token refresh');
        console.log('[ChatCache] 🎯 Performance benefit: Loaded', totalMessages, 'messages with 1 API call instead of', totalThreadsWithMessages, 'individual calls');
      } else {
        console.log('[ChatCache] ⚠️ No meaningful data received from bulk loading');
      }
      
    } catch (error) {
      console.error('[ChatCache] ❌ Error in bulk loading:', error);
      
      // Enhanced error logging for authentication issues
      if (error instanceof Error) {
        if (error.message.includes('Authentication failed') || error.message.includes('Session expired')) {
          console.error('[ChatCache] 🔐 Authentication error during bulk loading - user may need to log in again');
        } else {
          console.error('[ChatCache] 📡 Network or API error during bulk loading:', error.message);
        }
      }
      
      // Don't throw error - graceful degradation
      console.log('[ChatCache] 🔄 Bulk loading failed, will rely on individual thread loading');
      
    } finally {
      setIsLoading(false);
    }
  }, [saveToStorage]);

  // NEW: Load initial threads with pagination
  const loadInitialThreads = useCallback(async () => {
    if (!userEmail || threadsLoading) return;
    
    console.log('[ChatCache] 🔄 Loading initial threads (page 1)...');
    setThreadsLoading(true);
    
    try {
      const { authApiFetch } = await import('@/lib/api');
      const { getSession } = await import('next-auth/react');
      
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      const response = await authApiFetch<PaginatedChatThreadsResponse>(
        '/chat-threads?page=1&limit=10', 
        freshSession.id_token
      );
      
      console.log('[ChatCache] ✅ Loaded initial threads:', response);
      
      // Convert to ChatThreadMeta format
      const threadMetas: ChatThreadMeta[] = response.threads.map(t => ({
        thread_id: t.thread_id,
        latest_timestamp: t.latest_timestamp,
        run_count: t.run_count,
        title: t.title,
        full_prompt: t.full_prompt
      }));
      
      setThreadsState(threadMetas);
      setThreadsPage(1);
      setThreadsHasMore(response.has_more);
      setTotalThreadsCount(response.total_count);
      
      console.log('[ChatCache] 📊 Pagination state:', {
        loaded: threadMetas.length,
        total: response.total_count,
        hasMore: response.has_more
      });
      
      // IMPORTANT: Also load all conversation messages for the loaded threads
      // This maintains the original functionality where messages were loaded alongside threads
      console.log('[ChatCache] 🔄 Now loading all conversation messages for threads...');
      try {
        await loadAllMessagesFromAPI();
        console.log('[ChatCache] ✅ Successfully loaded all conversation messages alongside threads');
      } catch (messageError) {
        console.error('[ChatCache] ⚠️ Failed to load messages, but threads loaded successfully:', messageError);
        // Don't throw here - we still have threads loaded even if messages failed
      }
      
    } catch (error) {
      console.error('[ChatCache] ❌ Failed to load initial threads:', error);
      throw error; // Re-throw thread loading errors
    } finally {
      setThreadsLoading(false);
    }
  }, [userEmail, threadsLoading, loadAllMessagesFromAPI]);
  
  // NEW: Load more threads
  const loadMoreThreads = useCallback(async () => {
    if (!userEmail || threadsLoading || !threadsHasMore) return;
    
    console.log('[ChatCache] 🔄 Loading more threads (page', threadsPage + 1, ')...');
    setThreadsLoading(true);
    
    try {
      const { authApiFetch } = await import('@/lib/api');
      const { getSession } = await import('next-auth/react');
      
      const freshSession = await getSession();
      if (!freshSession?.id_token) {
        throw new Error('No authentication token available');
      }

      const nextPage = threadsPage + 1;
      const response = await authApiFetch<PaginatedChatThreadsResponse>(
        `/chat-threads?page=${nextPage}&limit=10`, 
        freshSession.id_token
      );
      
      console.log('[ChatCache] ✅ Loaded more threads:', response);
      
      // Convert to ChatThreadMeta format
      const newThreadMetas: ChatThreadMeta[] = response.threads.map(t => ({
        thread_id: t.thread_id,
        latest_timestamp: t.latest_timestamp,
        run_count: t.run_count,
        title: t.title,
        full_prompt: t.full_prompt
      }));
      
      // Append to existing threads
      setThreadsState(prev => [...prev, ...newThreadMetas]);
      setThreadsPage(nextPage);
      setThreadsHasMore(response.has_more);
      setTotalThreadsCount(response.total_count);
      
      console.log('[ChatCache] 📊 Updated pagination state:', {
        loaded: threads.length + newThreadMetas.length,
        total: response.total_count,
        hasMore: response.has_more,
        page: nextPage
      });
      
      // IMPORTANT: Load messages for the newly loaded threads
      // This ensures conversation messages are available when user clicks on new threads
      if (newThreadMetas.length > 0) {
        console.log('[ChatCache] 🔄 Loading conversation messages for', newThreadMetas.length, 'newly loaded threads...');
        try {
          // Reload all messages to include the new threads' messages
          await loadAllMessagesFromAPI();
          console.log('[ChatCache] ✅ Successfully loaded messages for newly loaded threads');
        } catch (messageError) {
          console.error('[ChatCache] ⚠️ Failed to load messages for new threads:', messageError);
          // Don't throw here - we still have the threads loaded
        }
      }
      
    } catch (error) {
      console.error('[ChatCache] ❌ Failed to load more threads:', error);
      throw error; // Re-throw thread loading errors
    } finally {
      setThreadsLoading(false);
    }
  }, [userEmail, threadsLoading, threadsHasMore, threadsPage, threads.length, loadAllMessagesFromAPI]);
  
  // NEW: Reset pagination
  const resetPagination = useCallback(() => {
    console.log('[ChatCache] 🔄 Resetting pagination...');
    setThreadsState([]);
    setThreadsPage(1);
    setThreadsHasMore(true);
    setTotalThreadsCount(0);
    setThreadsLoading(false);
  }, []);

  const contextValue: ChatCacheContextType = {
    // State
    threads,
    messages: currentMessages,
    activeThreadId,
    isLoading,
    
    // Pagination state
    threadsPage,
    threadsHasMore,
    threadsLoading,
    totalThreadsCount,
    
    // Actions
    setThreads,
    setMessages,
    setActiveThreadId,
    addMessage,
    updateMessage,
    addThread,
    removeThread,
    updateThread,
    
    // Pagination actions
    loadInitialThreads,
    loadMoreThreads,
    resetPagination,
    
    // Cache management
    invalidateCache,
    refreshFromAPI,
    isDataStale,
    setLoading,
    
    // NEW: Track if this is a page refresh
    isPageRefresh,
    
    // NEW: Force API refresh on F5
    forceAPIRefresh,
    
    // NEW: Check if messages exist for a specific thread
    hasMessagesForThread: (threadId: string) => {
      return !!messages[threadId] && messages[threadId].length > 0;
    },
    
    // NEW: Reset page refresh state
    resetPageRefresh,
    
    // NEW: Cross-tab loading state management
    isUserLoading,
    setUserLoadingState,
    checkUserLoadingState,
    setUserEmail: setUserEmailContext,
    
    // NEW: Clear cache for user changes (logout/login)
    clearCacheForUserChange,
    
    // NEW: Bulk load all messages, run-ids, and sentiments at once
    loadAllMessagesFromAPI,
    
    // NEW: Get cached run-ids and sentiments for a thread
    getRunIdsForThread: (threadId: string) => {
      return runIds[threadId] || [];
    },
    getSentimentsForThread: (threadId: string) => {
      return sentiments[threadId] || {};
    },
    
    // NEW: Update cached sentiment
    updateCachedSentiment: (threadId: string, runId: string, sentiment: boolean | null) => {
      console.log('[ChatCache] 📝 Updating cached sentiment:', threadId, runId, sentiment);
      setSentimentsState(prev => {
        const threadSentiments = { ...prev[threadId] };
        
        if (sentiment === null) {
          // Remove the sentiment entry
          delete threadSentiments[runId];
        } else {
          // Set the sentiment value
          threadSentiments[runId] = sentiment;
        }
        
        return {
          ...prev,
          [threadId]: threadSentiments
        };
      });
    }
  }

  return (
    <ChatCacheContext.Provider value={contextValue}>
      {children}
    </ChatCacheContext.Provider>
  )
}

export function useChatCache() {
  const context = useContext(ChatCacheContext)
  if (context === undefined) {
    throw new Error('useChatCache must be used within a ChatCacheProvider')
  }
  return context
} 