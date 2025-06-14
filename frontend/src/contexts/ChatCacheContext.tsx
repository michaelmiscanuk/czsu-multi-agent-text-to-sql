'use client'

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { ChatThreadMeta, ChatMessage } from '@/types'

interface CacheData {
  threads: ChatThreadMeta[];
  messages: { [threadId: string]: ChatMessage[] };
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
  
  // Actions
  setThreads: (threads: ChatThreadMeta[]) => void;
  setMessages: (threadId: string, messages: ChatMessage[]) => void;
  setActiveThreadId: (threadId: string | null) => void;
  addMessage: (threadId: string, message: ChatMessage) => void;
  updateMessage: (threadId: string, messageId: string, updatedMessage: ChatMessage) => void;
  addThread: (thread: ChatThreadMeta) => void;
  removeThread: (threadId: string) => void;
  updateThread: (threadId: string, updates: Partial<ChatThreadMeta>) => void;
  
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
}

const ChatCacheContext = createContext<ChatCacheContextType | undefined>(undefined)

// Cache configuration
const CACHE_KEY = 'czsu-chat-cache'
const ACTIVE_THREAD_KEY = 'czsu-last-active-chat'
const CACHE_DURATION = 5 * 60 * 1000 // 5 minutes in milliseconds
const PAGE_REFRESH_FLAG_KEY = 'czsu-page-refresh-flag'
const USER_LOADING_STATE_KEY = 'czsu-user-loading-state' // NEW: Cross-tab loading state

export function ChatCacheProvider({ children }: { children: React.ReactNode }) {
  // Internal state
  const [threads, setThreadsState] = useState<ChatThreadMeta[]>([])
  const [messages, setMessagesState] = useState<{ [threadId: string]: ChatMessage[] }>({})
  const [activeThreadId, setActiveThreadIdState] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<number>(0)
  const [userEmail, setUserEmail] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isPageRefresh, setIsPageRefresh] = useState(false)
  
  // NEW: Cross-tab loading state tied to user email
  const [isUserLoading, setIsUserLoading] = useState(false)
  
  // Track if this is the initial mount to detect page refresh
  const isInitialMount = useRef(true)
  const hasBeenHydrated = useRef(false)

  // Page refresh detection logic - fix to distinguish F5 from navigation
  useEffect(() => {
    // Ensure we're on the client side
    if (typeof window === 'undefined') return;
    
    console.log('[ChatCache] 🔄 Component mounted, checking if page refresh...');
    
    const now = Date.now();
    
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
      
      // Method 3: Check sessionStorage flag that persists only during session
      if (typeof sessionStorage !== 'undefined') {
        const refreshTimestamp = sessionStorage.getItem(PAGE_REFRESH_FLAG_KEY);
        
        if (refreshTimestamp) {
          const timeDiff = now - parseInt(refreshTimestamp, 10);
          // If less than 1 second since flag was set, it's likely a reload
          if (timeDiff < 1000) {
            return true;
          }
        }
      }
      
      return false;
    };

    // Check if localStorage has data (previous session)
    const hasLocalStorageData = typeof localStorage !== 'undefined' && !!localStorage.getItem(CACHE_KEY);
    
    // Final decision logic - be more conservative about detecting refresh
    const actualPageRefresh = isPageReload();
    
    // Only set page refresh if we're confident it's a real F5/reload
    const finalDecision = actualPageRefresh && hasLocalStorageData;
    
    console.log('[ChatCache] 🔍 Page refresh detection: ', {
      performanceNavType: typeof performance !== 'undefined' ? performance.navigation?.type : 'undefined',
      performanceEntries: typeof performance !== 'undefined' ? (performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming)?.type : 'undefined',
      sessionStorageFlag: typeof sessionStorage !== 'undefined' ? !!sessionStorage.getItem(PAGE_REFRESH_FLAG_KEY) : false,
      hasLocalStorageData,
      actualPageRefresh,
      finalDecision
    });
    
    setIsPageRefresh(finalDecision);
    
    // Set sessionStorage flag for next potential refresh detection
    if (typeof sessionStorage !== 'undefined') {
      sessionStorage.setItem(PAGE_REFRESH_FLAG_KEY, now.toString());
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
        activeThreadId,
        userEmail
      })
    }
  }, [threads, messages, activeThreadId, userEmail, saveToStorage])

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
    console.log('[ChatCache] 📝 Updating message:', messageId, 'in thread:', threadId)
    setMessagesState(prev => ({
      ...prev,
      [threadId]: (prev[threadId] || []).map(msg => 
        msg.id === messageId ? updatedMessage : msg
      )
    }))
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
    setActiveThreadIdState(null)
    setUserEmail(newUserEmail)
    setIsLoading(false)
    setIsPageRefresh(false)
    setIsUserLoading(false)
    
    console.log('[ChatCache] ✅ Cache state reset for user change');
  }, [userEmail]);

  // Get current messages for active thread
  const currentMessages = activeThreadId ? messages[activeThreadId] || [] : []

  // NEW: Functions for cross-tab loading state management
  const getUserLoadingKey = useCallback((email: string) => {
    return `${USER_LOADING_STATE_KEY}:${email}`;
  }, []);

  const setUserLoadingState = useCallback((email: string, loading: boolean) => {
    if (!email || typeof window === 'undefined' || typeof localStorage === 'undefined') return;
    
    try {
      const key = getUserLoadingKey(email);
      if (loading) {
        const loadingData = {
          email,
          loading: true,
          timestamp: Date.now(),
          tabId: Math.random().toString(36).substr(2, 9) // Unique tab identifier
        };
        localStorage.setItem(key, JSON.stringify(loadingData));
        console.log('[ChatCache] 🔒 User loading state set for:', email, 'across all tabs');
        
        // IMPORTANT: Also update local state immediately in current tab
        setIsUserLoading(true);
        
        // Force a custom storage event for current tab (since storage events don't fire in the same tab)
        window.dispatchEvent(new CustomEvent('userLoadingChange', { 
          detail: { email, loading: true } 
        }));
      } else {
        localStorage.removeItem(key);
        console.log('[ChatCache] 🔓 User loading state cleared for:', email, 'across all tabs');
        
        // IMPORTANT: Also update local state immediately in current tab
        setIsUserLoading(false);
        
        // Force a custom storage event for current tab
        window.dispatchEvent(new CustomEvent('userLoadingChange', { 
          detail: { email, loading: false } 
        }));
      }
    } catch (error) {
      console.error('[ChatCache] ❌ Error setting user loading state:', error);
    }
  }, [getUserLoadingKey]);

  const checkUserLoadingState = useCallback((email: string) => {
    if (!email || typeof window === 'undefined' || typeof localStorage === 'undefined') return false;
    
    try {
      const key = getUserLoadingKey(email);
      const stored = localStorage.getItem(key);
      
      if (stored) {
        const data = JSON.parse(stored);
        const age = Date.now() - data.timestamp;
        
        // Clear stale loading states (older than 5 minutes - prevent stuck states)
        if (age > 5 * 60 * 1000) {
          localStorage.removeItem(key);
          console.log('[ChatCache] 🧹 Cleared stale loading state for:', email);
          return false;
        }
        
        console.log('[ChatCache] 🔍 User loading state check for:', email, '- loading:', data.loading);
        return data.loading === true;
      }
      
      return false;
    } catch (error) {
      console.error('[ChatCache] ❌ Error checking user loading state:', error);
      return false;
    }
  }, [getUserLoadingKey]);

  // NEW: Better cross-tab loading state synchronization
  useEffect(() => {
    if (!userEmail || typeof window === 'undefined') return;

    // Handle storage events from OTHER tabs
    const handleStorageChange = (e: StorageEvent) => {
      if (!e.key || !e.key.startsWith(USER_LOADING_STATE_KEY)) return;
      
      // Check if this change affects the current user
      const expectedKey = getUserLoadingKey(userEmail);
      if (e.key !== expectedKey) return;
      
      console.log('[ChatCache] 📡 Cross-tab loading state change detected from ANOTHER tab for:', userEmail);
      
      // Update local state based on storage change
      const isCurrentlyLoading = checkUserLoadingState(userEmail);
      setIsUserLoading(isCurrentlyLoading);
    };

    // Handle custom events from CURRENT tab (since storage events don't fire in same tab)
    const handleCustomLoadingChange = (e: CustomEvent) => {
      if (e.detail.email === userEmail) {
        console.log('[ChatCache] 📡 Loading state change detected in CURRENT tab for:', userEmail, '- loading:', e.detail.loading);
        setIsUserLoading(e.detail.loading);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('userLoadingChange', handleCustomLoadingChange as EventListener);
    
    // Initial check when user email changes
    const initialLoadingState = checkUserLoadingState(userEmail);
    setIsUserLoading(initialLoadingState);
    console.log('[ChatCache] 🔍 Initial loading state check for:', userEmail, '- loading:', initialLoadingState);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('userLoadingChange', handleCustomLoadingChange as EventListener);
    };
  }, [userEmail, getUserLoadingKey, checkUserLoadingState]);

  const contextValue: ChatCacheContextType = {
    // State
    threads,
    messages: currentMessages,
    activeThreadId,
    isLoading,
    
    // Actions
    setThreads,
    setMessages,
    setActiveThreadId,
    addMessage,
    updateMessage,
    addThread,
    removeThread,
    updateThread,
    
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
    resetPageRefresh: () => {
      console.log('[ChatCache] 🔄 Resetting page refresh flag - future navigation will use cache');
      setIsPageRefresh(false);
    },
    
    // NEW: Cross-tab loading state management
    isUserLoading,
    setUserLoadingState,
    checkUserLoadingState,
    setUserEmail: setUserEmailContext,
    
    // NEW: Clear cache for user changes (logout/login)
    clearCacheForUserChange
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