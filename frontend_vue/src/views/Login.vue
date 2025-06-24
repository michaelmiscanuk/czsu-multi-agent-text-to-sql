<template>
  <div class="w-full max-w-md mx-auto bg-white rounded-2xl shadow-2xl border border-gray-100 min-h-[60vh] flex flex-col items-center justify-center p-8 mt-12">
    <div class="flex flex-col items-center mb-8">
      <div class="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-10 h-10 text-gray-400">
          <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118A7.5 7.5 0 0112 15.75a7.5 7.5 0 017.5 4.368" />
        </svg>
      </div>
      <h2 class="text-2xl font-bold mb-2">Login to Your Account</h2>
      <p class="text-gray-600 text-sm">Sign in to access the CZSU Data Explorer</p>
    </div>
    <div class="w-full flex flex-col items-center">
      <AuthButton />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { storeToRefs } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import AuthButton from '@/components/AuthButton.vue'

const router = useRouter()
const authStore = useAuthStore()
const { isAuthenticated } = storeToRefs(authStore)

// Redirect to chat if already authenticated
watch(() => authStore.isAuthenticated, (isAuthenticated) => {
  if (isAuthenticated) {
    router.replace('/chat')
  }
}, { immediate: true })

onMounted(() => {
  // Check authentication status on mount
  if (isAuthenticated.value) {
    router.replace('/chat')
  }
})
</script> 