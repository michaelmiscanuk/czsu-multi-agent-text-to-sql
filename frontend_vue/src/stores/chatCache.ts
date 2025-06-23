import { defineStore } from 'pinia';
import { ref, computed, watch } from 'vue';
import type { ChatThreadMeta, ChatMessage, PaginatedChatThreadsResponse, AnalyzeResponse } from '@/types';
import { authApiFetch } from '@/lib/api';

interface CacheData {
  threads: ChatThreadMeta[];
  messages: { [threadId: string]: ChatMessage[] };
  runIds: { [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] };
  sentiments: { [threadId: string]: { [runId: string]: boolean } };
  activeThreadId: string | null;
  lastUpdated: number;
  userEmail: string | null;
}

const CACHE_KEY = 'czsu-chat-cache';
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes
const PAGE_SIZE = 20;

export const useChatCacheStore = defineStore('chatCache', () => {
  // State
  const threads = ref<ChatThreadMeta[]>([]);
  const allMessages = ref<{ [threadId: string]: ChatMessage[] }>({});
  const runIds = ref<{ [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] }>({});
  const sentiments = ref<{ [threadId: string]: { [runId: string]: boolean } }>({});
  const activeThreadId = ref<string | null>(null);
  const isLoading = ref(false);
  const userEmail = ref<string | null>(null);
  
  // Pagination state
  const threadsPage = ref(1);
  const threadsHasMore = ref(true);
  const threadsLoading = ref(false);
  const totalThreadsCount = ref(0);
  
  // NEW: Cross-tab loading state management
  const isUserLoading = ref(false);
  
  // NEW: Track if this is a page refresh
  const isPageRefresh = ref(false);
  
  // Computed
  const messages = computed(() => {
    if (!activeThreadId.value) return [];
    return allMessages.value[activeThreadId.value] || [];
  });
  
  // Cache management
  const getCacheKey = () => `${CACHE_KEY}-${userEmail.value}`;
  
  const saveToCache = () => {
    if (!userEmail.value) return;
    
    const cacheData: CacheData = {
      threads: threads.value,
      messages: allMessages.value,
      runIds: runIds.value,
      sentiments: sentiments.value,
      activeThreadId: activeThreadId.value,
      lastUpdated: Date.now(),
      userEmail: userEmail.value,
    };
    
    try {
      localStorage.setItem(getCacheKey(), JSON.stringify(cacheData));
      console.log('[ChatCacheStore] 💾 Cache saved for user:', userEmail.value);
    } catch (error) {
      console.error('[ChatCacheStore] ❌ Failed to save cache:', error);
    }
  };
  
  const loadFromCache = (): boolean => {
    if (!userEmail.value) return false;
    
    try {
      const cached = localStorage.getItem(getCacheKey());
      if (!cached) return false;
      
      const data: CacheData = JSON.parse(cached);
      
      // Validate cache is for the correct user
      if (data.userEmail !== userEmail.value) {
        console.log('[ChatCacheStore] 👥 Cache user mismatch, clearing');
        localStorage.removeItem(getCacheKey());
        return false;
      }
      
      threads.value = data.threads || [];
      allMessages.value = data.messages || {};
      runIds.value = data.runIds || {};
      sentiments.value = data.sentiments || {};
      activeThreadId.value = data.activeThreadId;
      totalThreadsCount.value = data.threads?.length || 0;
      
      console.log('[ChatCacheStore] 📤 Cache loaded for user:', userEmail.value, '- threads:', threads.value.length);
      return true;
    } catch (error) {
      console.error('[ChatCacheStore] ❌ Failed to load cache:', error);
      return false;
    }
  };
  
  const isDataStale = (): boolean => {
    if (!userEmail.value) return true;
    
    try {
      const cached = localStorage.getItem(getCacheKey());
      if (!cached) return true;
      
      const data: CacheData = JSON.parse(cached);
      const age = Date.now() - (data.lastUpdated || 0);
      
      return age > CACHE_DURATION;
    } catch {
      return true;
    }
  };
  
  const invalidateCache = () => {
    if (!userEmail.value) return;
    localStorage.removeItem(getCacheKey());
    console.log('[ChatCacheStore] 🗑️ Cache invalidated for user:', userEmail.value);
  };
  
  const clearCacheForUserChange = (newUserEmail?: string | null) => {
    console.log('[ChatCacheStore] 👤 User change detected - clearing all cache data');
    
    // Clear all possible cache keys
    const keys = Object.keys(localStorage);
    keys.forEach(key => {
      if (key.startsWith(CACHE_KEY)) {
        localStorage.removeItem(key);
      }
    });
    
    // Reset state
    threads.value = [];
    allMessages.value = {};
    runIds.value = {};
    sentiments.value = {};
    activeThreadId.value = null;
    threadsPage.value = 1;
    threadsHasMore.value = true;
    totalThreadsCount.value = 0;
    
    if (newUserEmail) {
      userEmail.value = newUserEmail;
    }
    
    console.log('[ChatCacheStore] ✅ Cache cleared for user change');
  };
  
  // Cross-tab loading state management
  const setUserLoadingState = (email: string, loading: boolean) => {
    const key = `czsu-user-loading-${email}`;
    if (loading) {
      localStorage.setItem(key, Date.now().toString());
    } else {
      localStorage.removeItem(key);
    }
    
    if (email === userEmail.value) {
      isUserLoading.value = loading;
    }
    
    console.log('[ChatCacheStore] 🔒 User loading state set:', email, loading);
  };
  
  const checkUserLoadingState = (email: string): boolean => {
    const key = `czsu-user-loading-${email}`;
    const loadingTimestamp = localStorage.getItem(key);
    
    if (!loadingTimestamp) return false;
    
    // Check if loading state is stale (older than 10 minutes)
    const age = Date.now() - parseInt(loadingTimestamp);
    if (age > 600000) { // 10 minutes
      localStorage.removeItem(key);
      return false;
    }
    
    return true;
  };
  
  // API calls
  const loadInitialThreads = async () => {
    if (!userEmail.value) {
      console.log('[ChatCacheStore] ⚠ No user email for loadInitialThreads');
      return;
    }
    
    console.log('[ChatCacheStore] 🚀 Loading initial threads with pagination...');
    threadsLoading.value = true;
    
    try {
      // Get fresh session/token - this will need to be adapted based on auth system
      const token = localStorage.getItem('auth-token'); // Placeholder
      if (!token) {
        throw new Error('No authentication token available');
      }
      
      const response = await authApiFetch<PaginatedChatThreadsResponse>(
        `/chat/threads?page=1&limit=${PAGE_SIZE}`,
        token
      );
      
      threads.value = response.threads.map(thread => ({
        thread_id: thread.thread_id,
        latest_timestamp: thread.latest_timestamp,
        run_count: thread.run_count,
        title: thread.title,
        full_prompt: thread.full_prompt,
      }));
      
      totalThreadsCount.value = response.total_count;
      threadsPage.value = 1;
      threadsHasMore.value = response.has_more;
      
      console.log('[ChatCacheStore] ✅ Initial threads loaded:', threads.value.length, 'of', totalThreadsCount.value);
      
      // Load all messages in bulk
      await loadAllMessagesFromAPI();
      
      // Save to cache
      saveToCache();
      
    } catch (error) {
      console.error('[ChatCacheStore] ❌ Error loading initial threads:', error);
    } finally {
      threadsLoading.value = false;
    }
  };
  
  const loadMoreThreads = async () => {
    if (!userEmail.value || !threadsHasMore.value || threadsLoading.value) {
      return;
    }
    
    console.log('[ChatCacheStore] 📖 Loading more threads, page:', threadsPage.value + 1);
    threadsLoading.value = true;
    
    try {
      const token = localStorage.getItem('auth-token'); // Placeholder
      if (!token) {
        throw new Error('No authentication token available');
      }
      
      const response = await authApiFetch<PaginatedChatThreadsResponse>(
        `/chat/threads?page=${threadsPage.value + 1}&limit=${PAGE_SIZE}`,
        token
      );
      
      const newThreads = response.threads.map(thread => ({
        thread_id: thread.thread_id,
        latest_timestamp: thread.latest_timestamp,
        run_count: thread.run_count,
        title: thread.title,
        full_prompt: thread.full_prompt,
      }));
      
      threads.value = [...threads.value, ...newThreads];
      threadsPage.value = response.page;
      threadsHasMore.value = response.has_more;
      
      console.log('[ChatCacheStore] ✅ More threads loaded:', newThreads.length, 'total:', threads.value.length);
      
      // Save to cache
      saveToCache();
      
    } catch (error) {
      console.error('[ChatCacheStore] ❌ Error loading more threads:', error);
    } finally {
      threadsLoading.value = false;
    }
  };
  
  const loadAllMessagesFromAPI = async () => {
    if (!userEmail.value) {
      console.log('[ChatCacheStore] ⚠ No user email for loadAllMessagesFromAPI');
      return;
    }
    
    console.log('[ChatCacheStore] 📨 Loading all messages from API...');
    
    try {
      const token = localStorage.getItem('auth-token'); // Placeholder
      if (!token) {
        throw new Error('No authentication token available');
      }
      
      const response = await authApiFetch<{
        messages: { [threadId: string]: any[] };
        runIds: { [threadId: string]: { run_id: string; prompt: string; timestamp: string }[] };
        sentiments: { [threadId: string]: { [runId: string]: boolean } };
      }>('/chat/all-messages', token);
      
      // Convert API messages to ChatMessage format
      const convertedMessages: { [threadId: string]: ChatMessage[] } = {};
      
      Object.entries(response.messages).forEach(([threadId, messages]) => {
        convertedMessages[threadId] = messages.map(msg => ({
          id: msg.id || msg.message_id,
          threadId: threadId,
          user: msg.user || userEmail.value || 'unknown',
          content: msg.content || msg.message_content || '',
          isUser: msg.isUser !== undefined ? msg.isUser : msg.is_user || false,
          createdAt: msg.createdAt || msg.timestamp || Date.now(),
          error: msg.error,
          meta: msg.meta || {
            datasetsUsed: msg.top_selection_codes || [],
            sqlQuery: msg.sql || null,
            datasetUrl: msg.datasetUrl || null,
            runId: msg.run_id,
            topChunks: msg.top_chunks || []
          },
          queriesAndResults: msg.queries_and_results || [],
          isLoading: false,
          isError: msg.isError || false,
        }));
      });
      
      allMessages.value = convertedMessages;
      runIds.value = response.runIds || {};
      sentiments.value = response.sentiments || {};
      
      console.log('[ChatCacheStore] ✅ All messages loaded for', Object.keys(convertedMessages).length, 'threads');
      
    } catch (error) {
      console.error('[ChatCacheStore] ❌ Error loading all messages:', error);
    }
  };
  
  // Actions
  const setUserEmail = (email: string | null) => {
    userEmail.value = email;
  };
  
  const setLoading = (loading: boolean) => {
    isLoading.value = loading;
  };
  
  const setActiveThreadId = (threadId: string | null) => {
    activeThreadId.value = threadId;
    saveToCache();
  };
  
  const setThreads = (newThreads: ChatThreadMeta[]) => {
    threads.value = newThreads;
    saveToCache();
  };
  
  const setMessages = (threadId: string, messages: ChatMessage[]) => {
    allMessages.value = {
      ...allMessages.value,
      [threadId]: messages
    };
    saveToCache();
  };
  
  const addMessage = (threadId: string, message: ChatMessage) => {
    if (!allMessages.value[threadId]) {
      allMessages.value[threadId] = [];
    }
    allMessages.value[threadId].push(message);
    saveToCache();
  };
  
  const updateMessage = (threadId: string, messageId: string, updatedMessage: ChatMessage) => {
    if (!allMessages.value[threadId]) return;
    
    const index = allMessages.value[threadId].findIndex(msg => msg.id === messageId);
    if (index !== -1) {
      allMessages.value[threadId][index] = updatedMessage;
      saveToCache();
    }
  };
  
  const addThread = (thread: ChatThreadMeta) => {
    threads.value.unshift(thread);
    totalThreadsCount.value++;
    saveToCache();
  };
  
  const removeThread = (threadId: string) => {
    threads.value = threads.value.filter(t => t.thread_id !== threadId);
    delete allMessages.value[threadId];
    delete runIds.value[threadId];
    delete sentiments.value[threadId];
    totalThreadsCount.value = Math.max(0, totalThreadsCount.value - 1);
    saveToCache();
  };
  
  const updateThread = (threadId: string, updates: Partial<ChatThreadMeta>) => {
    const index = threads.value.findIndex(t => t.thread_id === threadId);
    if (index !== -1) {
      threads.value[index] = { ...threads.value[index], ...updates };
      saveToCache();
    }
  };
  
  const resetPagination = () => {
    threadsPage.value = 1;
    threadsHasMore.value = true;
    totalThreadsCount.value = 0;
  };
  
  const refreshFromAPI = async () => {
    invalidateCache();
    resetPagination();
    await loadInitialThreads();
  };
  
  const forceAPIRefresh = async () => {
    console.log('[ChatCacheStore] 🔄 Force API refresh (F5 detected)');
    isPageRefresh.value = true;
    await refreshFromAPI();
  };
  
  const hasMessagesForThread = (threadId: string): boolean => {
    return !!(allMessages.value[threadId] && allMessages.value[threadId].length > 0);
  };
  
  const resetPageRefresh = () => {
    isPageRefresh.value = false;
  };
  
  const getRunIdsForThread = (threadId: string) => {
    return runIds.value[threadId] || [];
  };
  
  const getSentimentsForThread = (threadId: string) => {
    return sentiments.value[threadId] || {};
  };
  
  const updateCachedSentiment = (threadId: string, runId: string, sentiment: boolean | null) => {
    if (!sentiments.value[threadId]) {
      sentiments.value[threadId] = {};
    }
    
    if (sentiment === null) {
      delete sentiments.value[threadId][runId];
    } else {
      sentiments.value[threadId][runId] = sentiment;
    }
    
    saveToCache();
  };
  
  // Auto-save cache when important state changes
  watch([threads, allMessages, activeThreadId], () => {
    if (userEmail.value) {
      saveToCache();
    }
  }, { deep: true });
  
  // Initialize from cache when user email is set
  watch(userEmail, (newEmail) => {
    if (newEmail) {
      const loaded = loadFromCache();
      if (!loaded) {
        console.log('[ChatCacheStore] 📋 No cache found for user:', newEmail);
      }
    }
  }, { immediate: true });
  
  // Detect page refresh
  if (typeof window !== 'undefined') {
    if (performance.navigation?.type === 1 || 
        (performance.getEntriesByType('navigation')[0] as any)?.type === 'reload') {
      isPageRefresh.value = true;
      console.log('[ChatCacheStore] 🔄 Page refresh detected');
    }
  }
  
  return {
    // State
    threads,
    messages,
    activeThreadId,
    isLoading,
    threadsPage,
    threadsHasMore,
    threadsLoading,
    totalThreadsCount,
    isUserLoading,
    isPageRefresh,
    
    // Actions
    setUserEmail,
    setLoading,
    setActiveThreadId,
    setThreads,
    setMessages,
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
    forceAPIRefresh,
    hasMessagesForThread,
    resetPageRefresh,
    setUserLoadingState,
    checkUserLoadingState,
    clearCacheForUserChange,
    loadAllMessagesFromAPI,
    getRunIdsForThread,
    getSentimentsForThread,
    updateCachedSentiment,
  };
}); 