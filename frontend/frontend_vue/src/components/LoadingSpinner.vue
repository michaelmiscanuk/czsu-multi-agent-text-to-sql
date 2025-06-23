<template>
  <div :class="`flex flex-col items-center justify-center py-4 ${className}`">
    <div 
      :class="`${sizeClasses[size]} border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-2`"
      role="status"
      aria-label="Loading"
    />
    <div 
      v-if="text"
      :class="`${textSizes[size]} text-gray-500 text-center`"
    >
      {{ text }}
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

const props = withDefaults(defineProps<Props>(), {
  size: 'md',
  className: ''
});

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8'
};

const textSizes = {
  sm: 'text-xs',
  md: 'text-sm',
  lg: 'text-base'
};
</script>

<style scoped>
/* Animation is already provided by Tailwind's animate-spin */
</style> 