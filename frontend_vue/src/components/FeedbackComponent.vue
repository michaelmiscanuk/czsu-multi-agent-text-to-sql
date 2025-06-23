<template>
  <div class="flex items-center space-x-2 relative">
    <!-- Show "selected" message if sentiment is already chosen -->
    <div v-if="currentSentiment !== null" class="flex items-center space-x-1 px-2 py-1 rounded bg-blue-50 text-blue-700 text-sm font-medium">
      <span>selected:</span>
      <span>{{ currentSentiment === true ? '👍' : '👎' }}</span>
    </div>
    
    <!-- Show clickable thumbs if no sentiment is selected -->
    <template v-else>
      <!-- Thumbs up -->
      <button
        class="p-1 rounded transition-colors text-gray-400 hover:text-blue-600 hover:bg-blue-50"
        title="Good response"
        @click="handleFeedback(1)"
      >
        👍
      </button>
      
      <!-- Thumbs down -->
      <button
        class="p-1 rounded transition-colors text-gray-400 hover:text-blue-600 hover:bg-blue-50"
        title="Poor response"
        @click="handleFeedback(0)"
      >
        👎
      </button>
    </template>
    
    <!-- Comment button with fixed positioning context -->
    <div class="relative">
      <button
        ref="commentButtonRef"
        :class="`p-1 rounded transition-colors ${
          showCommentBox 
            ? 'text-blue-600 bg-blue-50' 
            : hasProvidedComment
            ? 'text-green-600 hover:text-green-700 hover:bg-green-50'
            : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
        }`"
        :title="hasProvidedComment ? 'Comment provided - click to edit' : 'Add comment'"
        @click="showCommentBox = !showCommentBox"
      >
        <!-- Comment icon with checkmark overlay when comment provided -->
        <div v-if="hasProvidedComment" class="relative">
          <span>💬</span>
          <div class="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
            <span class="text-white text-xs leading-none">✓</span>
          </div>
        </div>
        <span v-else>💬</span>
      </button>
      
      <!-- Comment box - positioned relative to comment button wrapper -->
      <div 
        v-if="showCommentBox"
        ref="commentBoxRef"
        class="absolute bottom-full right-0 mb-2 p-3 bg-white border border-gray-200 rounded-lg shadow-lg min-w-[300px] z-20"
      >
        <textarea
          v-model="comment"
          placeholder="Share your feedback..."
          class="w-full p-2 border border-gray-300 rounded text-sm resize-none"
          rows="3"
          autoFocus
        />
        <div class="flex justify-end space-x-2 mt-2">
          <button
            class="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
            @click="showCommentBox = false"
          >
            Cancel
          </button>
          <button
            class="px-4 py-2 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            :disabled="!comment.trim()"
            @click="handleCommentSubmit"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Props {
  messageId: string
  runId?: string
  threadId: string
  feedbackState: { [key: string]: { feedback: number | null; comment?: string } }
  currentSentiment?: boolean | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  feedbackSubmit: [runId: string, feedback: number, comment?: string]
  commentSubmit: [runId: string, comment: string]
  sentimentUpdate: [runId: string, sentiment: boolean | null]
}>()

// State
const showCommentBox = ref(false)
const comment = ref('')
const hasProvidedComment = ref(false)
const commentButtonRef = ref<HTMLButtonElement>()
const commentBoxRef = ref<HTMLDivElement>()
const persistentFeedback = ref<number | null>(null)

// Computed
const messageFeedback = computed(() => 
  props.feedbackState[props.runId || props.messageId] || { feedback: null, comment: undefined }
)

// Use either API feedback state or persistent localStorage feedback
const effectiveFeedbackValue = computed(() => 
  messageFeedback.value.feedback !== null ? messageFeedback.value.feedback : persistentFeedback.value
)

// Load persisted feedback for this specific message on component mount
onMounted(() => {
  const fetchPersistedFeedback = () => {
    try {
      const storageKey = 'czsu-persistent-feedback'
      const savedFeedback = localStorage.getItem(storageKey)
      
      if (savedFeedback) {
        const feedbackData = JSON.parse(savedFeedback)
        // Check if we have feedback for this message
        if (feedbackData[props.messageId]) {
          const storedFeedback = feedbackData[props.messageId].feedbackValue
          console.log('[FEEDBACK-STORAGE] Found persisted feedback for message:', 
            { messageId: props.messageId, feedback: storedFeedback })
          persistentFeedback.value = storedFeedback
        }
      }
    } catch (err) {
      console.error('[FEEDBACK-STORAGE] Error loading persisted feedback for message:', err)
    }
  }
  
  fetchPersistedFeedback()
})

// Click outside to close comment box
onMounted(() => {
  const handleClickOutside = (event: MouseEvent) => {
    if (showCommentBox.value && 
        commentBoxRef.value && 
        commentButtonRef.value &&
        !commentBoxRef.value.contains(event.target as Node) &&
        !commentButtonRef.value.contains(event.target as Node)) {
      showCommentBox.value = false
    }
  }

  document.addEventListener('mousedown', handleClickOutside)
  
  onUnmounted(() => {
    document.removeEventListener('mousedown', handleClickOutside)
  })
})

// Save feedback to separate localStorage for persistence
const saveFeedbackToLocalStorage = (id: string, feedbackValue: number) => {
  try {
    // Use a separate localStorage key that won't be affected by cache invalidations
    const storageKey = 'czsu-persistent-feedback'
    
    // Get existing feedback data or initialize empty object
    const existingData = localStorage.getItem(storageKey)
    const feedbackData = existingData ? JSON.parse(existingData) : {}
    
    // Store feedback data with message ID and run ID (if available)
    feedbackData[props.messageId] = {
      feedbackValue, 
      timestamp: Date.now(),
      threadId: props.threadId,
      runId: props.runId || null
    }
    
    // Save back to localStorage
    localStorage.setItem(storageKey, JSON.stringify(feedbackData))
    
    // Update local state
    persistentFeedback.value = feedbackValue
    
    console.log('[FEEDBACK-STORAGE] Saved feedback to persistent localStorage:', 
      { messageId: props.messageId, runId: props.runId || null, feedbackValue })
  } catch (err) {
    console.error('[FEEDBACK-STORAGE] Error saving feedback to localStorage:', err)
  }
}

const handleFeedback = (feedback: number) => {
  // Use runId if available, otherwise fallback to messageId
  console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleFeedback - using runId:', props.runId, 'messageId:', props.messageId)
  
  // Save to separate localStorage for persistence
  saveFeedbackToLocalStorage(props.runId || props.messageId, feedback)
  
  // Update sentiment if we have a runId (new sentiment system)
  if (props.runId) {
    const sentiment = feedback === 1 ? true : false
    emit('sentimentUpdate', props.runId, sentiment)
  }
  
  // Call the original onFeedbackSubmit function (existing LangSmith feedback)
  emit('feedbackSubmit', props.runId || props.messageId, feedback)
  showCommentBox.value = false
  comment.value = ''
}

const handleCommentSubmit = () => {
  // Use runId if available, otherwise fallback to messageId
  console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleCommentSubmit - using runId:', props.runId, 'messageId:', props.messageId)
  const feedbackValue = messageFeedback.value.feedback !== null ? messageFeedback.value.feedback : 1
  
  // Save to localStorage along with comment
  saveFeedbackToLocalStorage(props.runId || props.messageId, feedbackValue)
  
  // Call the comment submit function from MessageArea
  if (props.runId && comment.value.trim()) {
    emit('commentSubmit', props.runId, comment.value.trim())
  }
  
  showCommentBox.value = false
  hasProvidedComment.value = true // Mark that a comment was provided
  comment.value = ''
}
</script> 