<template>
  <div ref="containerRef" class="flex-1 overflow-y-auto chat-scrollbar pr-4">
    <div class="max-w-5xl mx-auto min-h-full">
      <div v-if="messages.length === 0" class="flex items-center justify-center h-full min-h-[400px]">
        <div class="text-center">
          <div class="text-6xl mb-4">💬</div>
          <h3 class="text-xl font-semibold text-gray-700 mb-2">Start a conversation</h3>
          <p class="text-gray-500">Ask me about your data and I'll help you analyze it!</p>
        </div>
      </div>
      
      <div v-else>
        <div
          v-for="message in messages"
          :key="message.id"
          :class="`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-6`"
        >
          <div class="flex flex-col max-w-2xl w-full">
            <!-- Message Content -->
            <div
              :class="`transition-all duration-200 rounded-2xl px-6 py-4 w-full select-text shadow-lg group
                ${message.isUser
                  ? 'light-blue-theme font-semibold hover:shadow-xl'
                  : message.isError
                    ? 'bg-red-50 border border-red-200 text-red-800 hover:shadow-xl hover:border-red-300'
                    : 'bg-white border border-blue-100 text-gray-800 hover:shadow-xl hover:border-blue-200'}
              `"
              :style="{
                fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)',
                fontSize: '0.97rem',
                lineHeight: 1.6,
                wordBreak: 'break-word',
                whiteSpace: 'pre-line'
              }"
            >
              <div v-if="message.isLoading && !message.content" class="flex items-center space-x-3">
                <div class="w-5 h-5 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin"></div>
                <span class="text-gray-600">Analyzing your request...</span>
              </div>
              <div v-else>
                {{ message.content || '' }}
                <span v-if="!message.content" class="text-gray-400 text-xs italic">Waiting for response...</span>
              </div>
              
              <!-- Progress bar for loading messages -->
              <SimpleProgressBar
                v-if="message.isLoading && message.startedAt"
                :message-id="parseInt(message.id)"
                :started-at="message.startedAt"
              />
            </div>
            
            <!-- Dataset used and SQL button for AI answers -->
            <div
              v-if="!message.isUser && !message.isLoading && (message.selectionCode || message.meta?.datasetUrl || message.meta?.datasetsUsed?.length || message.meta?.sqlQuery || message.meta?.topChunks?.length)"
              class="mt-3 flex items-center justify-between flex-wrap"
              :style="{ fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)' }"
            >
              <div class="flex items-center space-x-3 flex-wrap">
                <!-- Show multiple dataset codes if available -->
                <div v-if="message.meta?.datasetsUsed && message.meta.datasetsUsed.length > 0" class="flex items-center space-x-2 flex-wrap">
                  <span class="text-xs text-gray-500 mr-1" style="margin-left: 1rem">
                    Dataset{{ message.meta.datasetsUsed.length > 1 ? 's' : '' }} used:
                  </span>
                  <router-link
                    v-for="(code, index) in message.meta.datasetsUsed"
                    :key="index"
                    :to="`/data?table=${encodeURIComponent(code)}`"
                    class="inline-block px-3 py-1 rounded-full bg-blue-50 text-blue-700 font-mono text-xs font-semibold hover:bg-blue-100 transition-all duration-150 shadow-sm border border-blue-100"
                    style="text-decoration: none"
                  >
                    {{ code }}
                  </router-link>
                </div>
                
                <!-- Fallback to old single dataset approach for backward compatibility -->
                <div v-else-if="message.selectionCode || message.meta?.datasetUrl">
                  <span class="text-xs text-gray-500 mr-1" style="margin-left: 1rem">Dataset used:</span>
                  <router-link
                    :to="`/data?table=${encodeURIComponent(message.selectionCode || message.meta?.datasetUrl.replace('/datasets/', ''))}`"
                    class="inline-block px-3 py-1 rounded-full bg-blue-50 text-blue-700 font-mono text-xs font-semibold hover:bg-blue-100 transition-all duration-150 shadow-sm border border-blue-100"
                    style="text-decoration: none"
                  >
                    {{ message.selectionCode || (message.meta?.datasetUrl ? message.meta.datasetUrl.replace('/datasets/', '') : '') }}
                  </router-link>
                </div>
                
                <button
                  v-if="message.meta?.sqlQuery"
                  class="px-4 py-1 rounded-full light-blue-theme text-xs font-bold transition-all duration-150"
                  @click="$emit('sqlClick', message.id)"
                >
                  SQL
                </button>
                
                <button
                  v-if="message.meta?.topChunks && message.meta.topChunks.length > 0"
                  class="px-4 py-1 rounded-full light-blue-theme text-xs font-bold transition-all duration-150"
                  @click="$emit('pdfClick', message.id)"
                >
                  PDF
                </button>
              </div>
              
              <!-- Feedback component aligned to the right -->
              <FeedbackComponent
                v-if="threadId"
                :message-id="message.id"
                :run-id="message.meta?.run_id || messageRunIds[message.id]"
                :thread-id="threadId"
                :feedback-state="feedbackState"
                :current-sentiment="getSentimentForRunId(message.meta?.run_id || messageRunIds[message.id] || '')"
                @feedback-submit="handleFeedbackSubmit"
                @comment-submit="handleCommentSubmit"
                @sentiment-update="updateSentiment"
              />
            </div>
            
            <!-- Show feedback component even when no datasets/SQL - for messages without metadata -->
            <div
              v-if="!message.isUser && !message.isLoading && threadId && !(message.selectionCode || message.meta?.datasetUrl || message.meta?.datasetsUsed?.length || message.meta?.sqlQuery || message.meta?.topChunks?.length)"
              class="mt-3 flex justify-end"
            >
              <FeedbackComponent
                :message-id="message.id"
                :run-id="message.meta?.run_id || messageRunIds[message.id]"
                :thread-id="threadId"
                :feedback-state="feedbackState"
                :current-sentiment="getSentimentForRunId(message.meta?.run_id || messageRunIds[message.id] || '')"
                @feedback-submit="handleFeedbackSubmit"
                @comment-submit="handleCommentSubmit"
                @sentiment-update="updateSentiment"
              />
            </div>
          </div>
        </div>
      </div>
      
      <!-- New Chat Button at bottom of scrollable content -->
      <div class="flex justify-center py-6">
        <button
          class="px-4 py-2 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="isAnyLoading || false"
          title="Start a new chat"
          @click="$emit('newChat')"
        >
          + New Chat
        </button>
      </div>
      
      <div ref="bottomRef" />
    </div>
    
    <!-- SQL Modal -->
    <Modal :open="!!openSQLModalForMsgId" @close="$emit('closeSQLModal')">
      <h2 class="text-lg font-bold mb-4">SQL Commands & Results</h2>
      <div class="max-h-[60vh] overflow-y-auto pr-2 chat-scrollbar">
        <div v-if="sqlModalMessage && sqlModalMessage.queriesAndResults && sqlModalMessage.queriesAndResults.length > 0" class="space-y-6">
          <div
            v-for="([sql, result], idx) in uniqueQueriesAndResults"
            :key="idx"
            class="bg-gray-50 rounded border border-gray-200 p-0"
          >
            <div class="bg-gray-100 px-4 py-2 rounded-t text-xs font-semibold text-gray-700 border-b border-gray-200">
              SQL Command {{ idx + 1 }}
            </div>
            <div class="p-3 font-mono text-xs whitespace-pre-line text-gray-900">
              <div v-for="(line, i) in sql.split('\n')" :key="i">
                {{ line }}
                <br v-if="i !== sql.split('\n').length - 1">
              </div>
            </div>
            <div class="bg-gray-100 px-4 py-2 text-xs font-semibold text-gray-700 border-t border-gray-200">Result</div>
            <div class="p-3 font-mono text-xs whitespace-pre-line text-gray-800">
              {{ typeof result === 'string' ? result : JSON.stringify(result, null, 2) }}
            </div>
          </div>
        </div>
        <div v-else class="text-gray-500">No SQL commands available.</div>
      </div>
    </Modal>
    
    <!-- PDF Modal -->
    <Modal :open="!!openPDFModalForMsgId" @close="$emit('closePDFModal')">
      <h2 class="text-lg font-bold mb-4">PDF Document Chunks</h2>
      <div class="max-h-[60vh] overflow-y-auto pr-2 chat-scrollbar">
        <div v-if="pdfModalMessage && pdfModalMessage.meta?.topChunks && pdfModalMessage.meta.topChunks.length > 0" class="space-y-6">
          <div
            v-for="(chunk, idx) in pdfModalMessage.meta.topChunks"
            :key="idx"
            class="bg-gray-50 rounded border border-gray-200 p-0"
          >
            <div class="bg-gray-100 px-4 py-2 rounded-t text-xs font-semibold text-gray-700 border-b border-gray-200">
              PDF Chunk {{ idx + 1 }}
              <span v-if="chunk.metadata?.source" class="ml-2 text-gray-600">({{ chunk.metadata.source }})</span>
            </div>
            <div class="p-3 text-xs whitespace-pre-line text-gray-800 leading-relaxed">
              {{ chunk.content }}
            </div>
          </div>
        </div>
        <div v-else class="text-gray-500">No PDF chunks available.</div>
      </div>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, computed, nextTick } from 'vue'
import { useSentiment } from '@/composables/useSentiment'
import { useChatCacheStore } from '@/stores/chatCache'
import { authApiFetch } from '@/lib/api'
import Modal from './Modal.vue'
import SimpleProgressBar from './SimpleProgressBar.vue'
import FeedbackComponent from './FeedbackComponent.vue'
import type { ChatMessage } from '@/types'
import { useAuthStore } from '@/stores/auth'

// Props
interface Props {
  messages: any[]
  threadId: string | null
  openSQLModalForMsgId: string | null
  openPDFModalForMsgId: string | null
  isLoading: boolean
  isAnyLoading?: boolean
  threads?: any[]
  activeThreadId?: string | null
}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  sqlClick: [msgId: string]
  closeSQLModal: []
  pdfClick: [msgId: string]
  closePDFModal: []
  newChat: []
}>()

// Refs
const bottomRef = ref<HTMLDivElement>()
const containerRef = ref<HTMLDivElement>()

// State
const feedbackState = ref<{ [runId: string]: { feedback: number | null; comment?: string } }>({})
const messageRunIds = ref<{[messageId: string]: string}>({})
const langsmithFeedbackSent = ref<Set<string>>(new Set())

// Stores and composables
const chatCacheStore = useChatCacheStore()
const { sentiments, updateSentiment, loadSentiments, getSentimentForRunId } = useSentiment()
const authStore = useAuthStore()

// Computed properties
const sqlModalMessage = computed(() => {
  return props.messages.find(m => m.id === props.openSQLModalForMsgId)
})

const pdfModalMessage = computed(() => {
  return props.messages.find(m => m.id === props.openPDFModalForMsgId)
})

const uniqueQueriesAndResults = computed(() => {
  if (!sqlModalMessage.value?.queriesAndResults) return []
  
  const uniqueMap = new Map()
  sqlModalMessage.value.queriesAndResults.forEach(([q, r]: [string, string]) => {
    uniqueMap.set(q, [q, r])
  })
  return Array.from(uniqueMap.values()) as [string, string][]
})

// Auto-scroll to bottom when messages change or thread changes
watch([() => props.messages, () => props.threadId], async () => {
  await nextTick()
  if (bottomRef.value) {
    bottomRef.value.scrollIntoView({ behavior: 'smooth' })
  }
})

// Reset LangSmith feedback tracking when thread changes
watch(() => props.threadId, () => {
  langsmithFeedbackSent.value = new Set()
})

// Load run-ids from cache instead of making API calls
watch(() => props.threadId, () => {
  const loadRunIdsFromCache = () => {
    if (!props.threadId) return
    
    console.log('[FEEDBACK-DEBUG] Loading run_ids from cache for thread:', props.threadId)
    
    // Get cached run-ids for this thread
    const cachedRunIds = chatCacheStore.getRunIdsForThread(props.threadId)
    console.log('[FEEDBACK-DEBUG] Found cached run_ids:', cachedRunIds.length)
    
    if (cachedRunIds && cachedRunIds.length > 0) {
      const newMessageRunIds: {[messageId: string]: string} = {}
      
      // Get all non-user messages (AI responses) in order
      const aiMessages = props.messages.filter(message => !message.isUser)
      
      // For each AI message, try to find a matching run_id
      aiMessages.forEach((message, index) => {
        // Skip if message already has run_id in meta
        if (message.meta?.runId) {
          console.log('[FEEDBACK-DEBUG] Message already has run_id in meta:', message.meta.runId)
          newMessageRunIds[message.id] = message.meta.runId
          return
        }
        
        // Try to match by order (most recent AI message gets most recent run_id)
        if (index < cachedRunIds.length) {
          const runIdEntry = cachedRunIds[cachedRunIds.length - 1 - index] // Reverse order
          if (runIdEntry) {
            newMessageRunIds[message.id] = runIdEntry.run_id
            console.log('[FEEDBACK-DEBUG] Matched message by order:', {
              messageId: message.id,
              runId: runIdEntry.run_id,
              messageIndex: index
            })
          }
        }
      })
      
      messageRunIds.value = newMessageRunIds
      console.log('[FEEDBACK-DEBUG] Final messageRunIds mapping:', newMessageRunIds)
    }
  }
  
  loadRunIdsFromCache()
})

// Load sentiments from cache when thread changes
watch(() => props.threadId, () => {
  if (props.threadId) {
    // Get cached sentiments for this thread from ChatCacheContext
    const cachedSentiments = chatCacheStore.getSentimentsForThread(props.threadId)
    console.log('[SENTIMENT-DEBUG] Loading sentiments from cache for thread:', props.threadId, cachedSentiments)
    
    if (cachedSentiments && Object.keys(cachedSentiments).length > 0) {
      // Load sentiments into the useSentiment hook
      loadSentiments(cachedSentiments)
    }
  }
})

// Feedback handlers
const handleFeedbackSubmit = async (runId: string, feedback: number, comment?: string) => {
  console.log('[FEEDBACK-DEBUG] MessageArea.handleFeedbackSubmit called with:', { runId, feedback, comment })
  
  // Prevent duplicate submissions to LangSmith
  if (langsmithFeedbackSent.value.has(runId)) {
    console.log('[FEEDBACK-DEBUG] Feedback already sent to LangSmith for runId:', runId)
    return
  }
  
  try {
    // Update local state immediately for UI responsiveness
    feedbackState.value[runId] = { 
      feedback, 
      ...(comment && { comment })
    }
    
    // Send feedback to LangSmith API
    const feedbackPayload = {
      run_id: runId,
      key: 'user_feedback',
      score: feedback, // 0 or 1
      ...(comment && { comment })
    }
    
    console.log('[FEEDBACK-DEBUG] Sending feedback to LangSmith API:', feedbackPayload)
    
    // Get auth token first
    const token = await authStore.getValidToken()
    if (!token) {
      throw new Error('No authentication token available')
    }
    
    const response = await authApiFetch('/api/langsmith-feedback', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: feedbackPayload
    })
    
    console.log('[FEEDBACK-DEBUG] LangSmith feedback API response:', response)
    
    // Mark as sent to prevent duplicates
    langsmithFeedbackSent.value.add(runId)
    
  } catch (error) {
    console.error('[FEEDBACK-DEBUG] Error submitting feedback to LangSmith:', error)
    // Don't remove from local state - user sees their feedback even if API fails
  }
}

const handleCommentSubmit = async (runId: string, comment: string) => {
  console.log('[FEEDBACK-DEBUG] MessageArea.handleCommentSubmit called with:', { runId, comment })
  
  try {
    // Update local state immediately
    const currentFeedback = feedbackState.value[runId]?.feedback || null
    feedbackState.value[runId] = { 
      feedback: currentFeedback, 
      ...(comment && { comment })
    }
    
    // Send comment to LangSmith API (as feedback with comment)
    const feedbackPayload = {
      run_id: runId,
      key: 'user_comment',
      score: currentFeedback !== null ? currentFeedback : 1, // Default to positive if no explicit feedback
      comment
    }
    
    console.log('[FEEDBACK-DEBUG] Sending comment to LangSmith API:', feedbackPayload)
    
    // Get auth token first
    const token = await authStore.getValidToken()
    if (!token) {
      throw new Error('No authentication token available')
    }
    
    const response = await authApiFetch('/api/langsmith-feedback', token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      data: feedbackPayload
    })
    
    console.log('[FEEDBACK-DEBUG] LangSmith comment API response:', response)
    
  } catch (error) {
    console.error('[FEEDBACK-DEBUG] Error submitting comment to LangSmith:', error)
    // Don't remove from local state - user sees their comment even if API fails
  }
}
</script> 