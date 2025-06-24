<template>
  <form @submit="handleSubmit" class="p-4 bg-white">
    <div class="flex items-center bg-[#F9F9F5] rounded-full p-3 shadow-md border border-gray-200 relative focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 focus-within:z-10 transition-all duration-200">
      <input
        ref="inputRef"
        type="text"
        placeholder="Type a message"
        aria-label="Type a message"
        :value="currentMessage"
        @input="handleChange"
        @keydown="handleKeydown"
        class="flex-grow px-4 py-2 bg-transparent focus:outline-none text-gray-700"
        :disabled="isLoading || false"
      />
      <button
        type="submit"
        aria-label="Send message"
        class="bg-gradient-to-r from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 rounded-full p-3 ml-2 shadow-md transition-all duration-200 group relative z-0 disabled:opacity-50 disabled:cursor-not-allowed"
        :disabled="isLoading || !currentMessage.trim()"
      >
        <svg class="w-6 h-6 text-white transform rotate-45 group-hover:scale-110 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
        </svg>
      </button>
    </div>
  </form>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface Props {
  currentMessage: string
  isLoading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  isLoading: false
})

const emit = defineEmits<{
  'update:currentMessage': [message: string]
  submit: [e: Event]
}>()

const inputRef = ref<HTMLInputElement>()

const handleChange = (e: Event) => {
  const target = e.target as HTMLInputElement
  emit('update:currentMessage', target.value)
}

const handleSubmit = (e: Event) => {
  e.preventDefault()
  if (!props.currentMessage.trim() || props.isLoading) return
  emit('submit', e)
}

const handleKeydown = (e: KeyboardEvent) => {
  // Allow Shift+Enter for new lines, Enter for submit
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSubmit(e)
  }
}

// Expose the input ref for parent access
defineExpose({
  inputRef,
  focus: () => inputRef.value?.focus()
})
</script> 