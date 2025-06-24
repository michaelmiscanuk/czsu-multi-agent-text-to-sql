<template>
  <div class="w-full mt-3">
    <div class="flex justify-between items-center mb-1">
      <span class="text-xs text-gray-500">Processing...</span>
      <span class="text-xs text-gray-500">
        {{ remainingMs > 0 ? (
          remainingMinutes > 0 ? 
            `~${remainingMinutes}m ${remainingSeconds}s remaining` : 
            `~${remainingSeconds}s remaining`
        ) : (
          'Completing...'
        ) }}
      </span>
    </div>
    <div class="h-[3px] w-full bg-gray-200 rounded-full overflow-hidden">
      <div
        class="h-[3px] bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-1000"
        :style="{ width: `${progress}%` }"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'

const PROGRESS_DURATION = 480000 // 8 minutes - matches backend analysis timeout

interface Props {
  messageId: number
  startedAt: number
}

const props = defineProps<Props>()

const progress = ref(0)

const intervalRef = ref<NodeJS.Timeout | null>(null)

const updateProgress = (percent: number) => {
  progress.value = percent
}

const update = () => {
  const elapsed = Date.now() - props.startedAt
  const percent = Math.min(95, (elapsed / PROGRESS_DURATION) * 100) // Cap at 95% until actual completion
  progress.value = percent
  
  // Don't auto-complete the progress bar - let the actual response completion do that
  if (elapsed >= PROGRESS_DURATION && intervalRef.value) {
    clearInterval(intervalRef.value)
  }
}

onMounted(() => {
  update()
  intervalRef.value = setInterval(update, 1000) // Update every second instead of every 100ms
})

onUnmounted(() => {
  if (intervalRef.value) clearInterval(intervalRef.value)
})

// Calculate estimated time remaining
const elapsed = computed(() => Date.now() - props.startedAt)
const remainingMs = computed(() => Math.max(0, PROGRESS_DURATION - elapsed.value))
const remainingMinutes = computed(() => Math.ceil(remainingMs.value / 60000))
const remainingSeconds = computed(() => Math.ceil((remainingMs.value % 60000) / 1000))
</script> 