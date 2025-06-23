<template>
  <div class="unified-white-block-system">
    <!-- Sidebar with its own scroll -->
    <aside class="w-64 bg-white border-r border-gray-200 flex flex-col">
      <!-- Sidebar Header -->
      <div class="flex items-center justify-between p-4 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
        <span class="font-bold text-lg text-blue-700">Chats</span>
        <button
          class="px-3 py-1.5 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          title="New chat"
          :disabled="isAnyLoading || !userEmail"
          @click="handleNewChat"
        >
          + New Chat
        </button>
      </div>
      
      <!-- Sidebar Chat List with Scroll -->
      <div ref="sidebarRef" class="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-1 chat-scrollbar">
        <div v-if="threadsLoading && threads.length === 0" class="text-center py-8">
          <div class="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
          <div class="text-sm text-gray-500">Loading your chats...</div>
        </div>
        
        <div v-else-if="threads.length === 0" class="text-center py-8">
          <div class="text-sm text-gray-500 mb-2">No chats yet</div>
          <div class="text-xs text-gray-400">Click "New Chat" to start</div>
        </div>
        
        <template v-else>
          <div v-for="thread in threads" :key="thread.thread_id" class="group">
            <input
              v-if="editingTitleId === thread.thread_id"
              v-model="newTitle"
              class="w-full px-3 py-2 text-sm rounded-lg bg-white border border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
              autoFocus
              @blur="handleRename(thread.thread_id, newTitle)"
              @keydown.enter="handleRename(thread.thread_id, newTitle)"
            />
            
            <div v-else class="flex items-center min-w-0">
              <button
                :class="`flex-1 text-left text-sm px-3 py-2 font-semibold rounded-lg transition-all duration-200 cursor-pointer min-w-0 ${
                  activeThreadId === thread.thread_id
                    ? 'font-extrabold light-blue-theme'
                    : 'text-[#181C3A]/80 hover:text-gray-300 hover:bg-gray-100'
                }`"
                :style="{ fontFamily: 'var(--font-inter)' }"
                :title="`${thread.full_prompt || thread.title || 'New Chat'}${thread.full_prompt && thread.full_prompt.length === 50 ? '...' : ''}`"
                @click="chatCacheStore.setActiveThreadId(thread.thread_id)"
                @dblclick="() => { editingTitleId = thread.thread_id; newTitle = thread.title || ''; }"
              >
                <div class="truncate block leading-tight">{{ thread.title || 'New Chat' }}</div>
              </button>
              <button
                class="flex-shrink-0 ml-1 text-gray-400 hover:text-red-500 text-lg font-bold px-2 py-1 rounded transition-colors"
                title="Delete chat"
                @click="handleDelete(thread.thread_id)"
              >
                ×
              </button>
            </div>
          </div>
          
          <!-- Infinite Scroll Loading Indicator -->
          <div v-if="threadsHasMore" ref="threadsObserverRef" class="w-full">
            <LoadingSpinner 
              size="sm" 
              text="Loading more chats..." 
              class="py-4"
            />
          </div>
          
          <!-- Loading indicator for additional pages -->
          <LoadingSpinner 
            v-if="threadsLoading && threads.length > 0"
            size="sm" 
            text="Loading more chats..." 
            class="py-2"
          />
          
          <!-- End of list indicator -->
          <div v-if="!threadsHasMore && threads.length > 10" class="text-center py-4">
            <div class="text-xs text-gray-400">
              All {{ totalThreadsCount }} chats loaded
            </div>
          </div>
        </template>
      </div>
    </aside>

    <!-- Main Chat Container -->
    <div class="flex-1 flex flex-col bg-gradient-to-br from-white to-blue-50/30 relative">
      <!-- Chat Messages Area with its own scroll -->
      <div class="flex-1 overflow-hidden">
        <MessageArea
          :messages="messages"
          :thread-id="activeThreadId"
          :open-s-q-l-modal-for-msg-id="openSQLModalForMsgId"
          :open-p-d-f-modal-for-msg-id="openPDFModalForMsgId"
          :is-loading="isAnyLoading"
          :is-any-loading="isAnyLoading"
          :threads="threads"
          :active-thread-id="activeThreadId"
          @sql-click="handleSQLButtonClick"
          @close-s-q-l-modal="handleCloseSQLModal"
          @pdf-click="handlePDFButtonClick"
          @close-p-d-f-modal="handleClosePDFModal"
          @new-chat="handleNewChat"
        />
      </div>
      
      <!-- Stationary Input Field -->
      <div class="bg-white border-t border-gray-200 shadow-lg">
        <form class="p-4 flex items-start gap-3 max-w-4xl mx-auto" @submit="handleSend">
          <textarea
            ref="inputRef"
            v-model="currentMessage"
            placeholder="Type your message here... (SHIFT+ENTER for new line)"
            class="flex-1 px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 bg-gray-50 transition-all duration-200 resize-none min-h-[48px] max-h-[200px]"
            :disabled="isAnyLoading"
            rows="1"
            :style="{
              height: 'auto',
              minHeight: '48px',
              maxHeight: '200px'
            }"
            @input="handleTextareaInput"
            @keydown="handleKeyDown"
          />
          <button
            type="submit"
            class="px-6 py-3 light-blue-theme rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 mt-1"
            :disabled="isAnyLoading || !currentMessage.trim()"
          >
            <span v-if="isAnyLoading" class="flex items-center gap-2">
              <div class="w-4 h-4 border-2 border-gray-400 border-t-gray-600 rounded-full animate-spin"></div>
              Sending...
            </span>
            <span v-else>Send</span>
          </button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { v4 as uuidv4 } from 'uuid'
import { useAuthStore } from '@/stores/auth'
import { useChatCacheStore } from '@/stores/chatCache'
import { useInfiniteScroll } from '@/composables/useInfiniteScroll'
import { authApiFetch, API_CONFIG } from '@/lib/api'
import MessageArea from '@/components/MessageArea.vue'
import LoadingSpinner from '@/components/LoadingSpinner.vue'
import type { ChatThreadMeta, ChatMessage, AnalyzeResponse, ChatThreadResponse } from '@/types'

// Interface for messages (matching React version)
interface Message {
  id: number
  content: string
  isUser: boolean
  type: string
  isLoading?: boolean
  startedAt?: number
  isError?: boolean
  selectionCode?: string | null
  queriesAndResults?: [string, string][]
  meta?: {
    datasetUrl?: string
    datasetsUsed?: string[]
    sqlQuery?: string
    run_id?: string
    topChunks?: Array<{
      content: string
      metadata: Record<string, any>
    }>
  }
}

const INITIAL_MESSAGE = [
  {
    id: 1,
    content: 'Hi there, how can I help you?',
    isUser: false,
    type: 'message'
  }
]

// Stores and composables
const router = useRouter()
const authStore = useAuthStore()
const chatCacheStore = useChatCacheStore()

// Extract all the complex state from the store
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
} = chatCacheStore

// User info
const userEmail = computed(() => authStore.user?.email || null)

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
    await loadMoreThreads()
  },
  { threshold: 1.0, rootMargin: '100px' }
)

// Local component state
const currentMessage = ref('')
const isLoading = ref(false)
const editingTitleId = ref<string | null>(null)
const newTitle = ref('')
const openSQLModalForMsgId = ref<string | null>(null)
const openPDFModalForMsgId = ref<string | null>(null)
const iteration = ref(0)
const maxIterations = ref(2)

// Combined loading state
const isAnyLoading = computed(() => isLoading.value || cacheLoading.value || isUserLoading.value)

// Refs for DOM elements
const inputRef = ref<HTMLTextAreaElement>()
const sidebarRef = ref<HTMLDivElement>()

// Simple text persistence
const setCurrentMessageWithPersistence = (message: string) => {
  currentMessage.value = message
  localStorage.setItem('czsu-draft-message', message)
}

// Event handlers
const handleSQLButtonClick = (msgId: string) => {
  openSQLModalForMsgId.value = msgId
}

const handleCloseSQLModal = () => {
  openSQLModalForMsgId.value = null
}

const handlePDFButtonClick = (msgId: string) => {
  openPDFModalForMsgId.value = msgId
}

const handleClosePDFModal = () => {
  openPDFModalForMsgId.value = null
}

const handleNewChat = async () => {
  if (!userEmail.value) {
    console.log('[ChatPage-newChat] ⚠ No user email available')
    return
  }

  try {
    console.log('[ChatPage-newChat] 🆕 Creating new chat...')
    
    // Create a new thread locally first for immediate UI response
    const newThreadId = uuidv4()
    const newThread: ChatThreadMeta = {
      thread_id: newThreadId,
      title: null,
      user_email: userEmail.value,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      full_prompt: null,
      message_count: 0
    }
    
    // Add the thread to the store
    addThread(newThread)
    
    // Set it as active immediately
    setActiveThreadId(newThreadId)
    
    // Clear current message input
    setCurrentMessageWithPersistence('')
    
    // Set initial messages
    setMessages([...INITIAL_MESSAGE])
    
    console.log('[ChatPage-newChat] ✅ New chat created with ID:', newThreadId)
  } catch (error) {
    console.error('[ChatPage-newChat] ❌ Error creating new chat:', error)
  }
}

const handleRename = async (threadId: string, title: string) => {
  if (!title.trim()) {
    editingTitleId.value = null
    return
  }

  try {
    // Update locally first
    updateThread(threadId, { title: title.trim() })
    
    // Then update on server
    const response = await authApiFetch(`${API_CONFIG.BASE_URL}/api/threads/${threadId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: title.trim() })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    console.log('[ChatPage-rename] ✅ Thread renamed:', threadId, title.trim())
  } catch (error) {
    console.error('[ChatPage-rename] ❌ Error renaming thread:', error)
  } finally {
    editingTitleId.value = null
    newTitle.value = ''
  }
}

const handleDelete = async (threadId: string) => {
  if (!confirm('Are you sure you want to delete this chat?')) {
    return
  }

  try {
    console.log('[ChatPage-delete] 🗑️ Deleting thread:', threadId)
    
    // Delete from server
    const response = await authApiFetch(`${API_CONFIG.BASE_URL}/api/threads/${threadId}`, {
      method: 'DELETE'
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    // Remove from local state
    removeThread(threadId)
    
    // If this was the active thread, clear active state
    if (activeThreadId.value === threadId) {
      setActiveThreadId(null)
      setMessages([...INITIAL_MESSAGE])
    }
    
    console.log('[ChatPage-delete] ✅ Thread deleted successfully:', threadId)
  } catch (error) {
    console.error('[ChatPage-delete] ❌ Error deleting thread:', error)
  }
}

const handleSend = async (e: Event) => {
  e.preventDefault()
  
  if (!currentMessage.value.trim() || isAnyLoading.value || !userEmail.value) {
    return
  }

  const messageText = currentMessage.value.trim()
  let currentThreadId = activeThreadId.value

  try {
    console.log('[ChatPage-send] 📤 Sending message:', messageText.substring(0, 50) + '...')
    
    // Create new thread if none exists
    if (!currentThreadId) {
      currentThreadId = uuidv4()
      const newThread: ChatThreadMeta = {
        thread_id: currentThreadId,
        title: null,
        user_email: userEmail.value,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        full_prompt: messageText.substring(0, 50),
        message_count: 0
      }
      
      addThread(newThread)
      setActiveThreadId(currentThreadId)
    }

    // Clear input immediately
    setCurrentMessageWithPersistence('')
    
    // Add user message
    const userMessage: Message = {
      id: Date.now(),
      content: messageText,
      isUser: true,
      type: 'message'
    }
    
    addMessage(userMessage)
    
    // Add loading AI message
    const loadingMessage: Message = {
      id: Date.now() + 1,
      content: '',
      isUser: false,
      type: 'message',
      isLoading: true,
      startedAt: Date.now()
    }
    
    addMessage(loadingMessage)
    
    // Start loading state
    setLoading(true)
    
    // Send to API
    const response = await authApiFetch(`${API_CONFIG.BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: messageText,
        thread_id: currentThreadId,
        user_email: userEmail.value
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result: AnalyzeResponse = await response.json()
    
    // Update the loading message with the response
    const aiMessage: Message = {
      id: loadingMessage.id,
      content: result.answer || 'No response received.',
      isUser: false,
      type: 'message',
      isLoading: false,
      selectionCode: result.selection_code,
      queriesAndResults: result.queries_and_results || [],
      meta: {
        datasetUrl: result.dataset_url,
        datasetsUsed: result.datasets_used,
        sqlQuery: result.sql_query,
        run_id: result.run_id,
        topChunks: result.top_chunks
      }
    }
    
    updateMessage(loadingMessage.id.toString(), aiMessage)
    
    console.log('[ChatPage-send] ✅ Message sent and response received')
    
  } catch (error) {
    console.error('[ChatPage-send] ❌ Error sending message:', error)
    
    // Update loading message to show error
    const errorMessage = {
      id: Date.now() + 1,
      content: 'Sorry, there was an error processing your request. Please try again.',
      isUser: false,
      type: 'message',
      isLoading: false,
      isError: true
    }
    
    updateMessage((Date.now() + 1).toString(), errorMessage)
  } finally {
    setLoading(false)
  }
}

const handleKeyDown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend(e)
  }
}

const handleTextareaInput = (e: Event) => {
  const target = e.target as HTMLTextAreaElement
  setCurrentMessageWithPersistence(target.value)
  
  // Auto-resize textarea based on content
  target.style.height = 'auto'
  target.style.height = Math.min(target.scrollHeight, 200) + 'px'
}

// Lifecycle hooks
onMounted(async () => {
  console.log('[ChatPage] 🚀 Component mounted')
  
  // Restore draft message
  const draftMessage = localStorage.getItem('czsu-draft-message')
  if (draftMessage) {
    currentMessage.value = draftMessage
  }
  
  // Load threads if user is authenticated
  if (userEmail.value) {
    await loadInitialThreads()
  }
})

// Watchers
watch(threadsHasMore, (newHasMore) => {
  setInfiniteScrollHasMore(newHasMore)
})

watch(infiniteScrollError, (error) => {
  if (error) {
    console.error('[ChatPage] Infinite scroll error:', error)
    setInfiniteScrollError(null)
  }
})

watch(activeThreadId, async (newThreadId, oldThreadId) => {
  if (newThreadId && newThreadId !== oldThreadId) {
    const hasCachedMessages = hasMessagesForThread(newThreadId)
    
    if (!hasCachedMessages) {
      try {
        setLoading(true)
        await loadAllMessagesFromAPI(newThreadId)
      } catch (error) {
        console.error('[ChatPage] Error loading messages:', error)
      } finally {
        setLoading(false)
      }
    }
  }
})

watch(userEmail, async (newEmail, oldEmail) => {
  if (newEmail !== oldEmail) {
    if (newEmail) {
      setUserEmail(newEmail)
      await loadInitialThreads()
    } else {
      clearCacheForUserChange()
    }
  }
})
</script> 