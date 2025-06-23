<template>
  <div v-if="open" class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
    <div class="bg-white rounded-lg shadow-lg max-w-2xl w-full p-6 relative">
      <button
        class="absolute top-2 right-2 text-gray-400 hover:text-gray-700 text-2xl font-bold"
        @click="onClose"
        title="Close"
      >
        ×
      </button>
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { watch } from 'vue';

interface Props {
  open: boolean;
  onClose: () => void;
}

const props = defineProps<Props>();

// Handle Escape key
watch(() => props.open, (isOpen) => {
  if (!isOpen) return;
  
  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      props.onClose();
    }
  };
  
  window.addEventListener('keydown', handleKeyDown);
  
  return () => {
    window.removeEventListener('keydown', handleKeyDown);
  };
});
</script> 