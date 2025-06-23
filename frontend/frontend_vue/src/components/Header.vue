<template>
  <header class="relative flex items-center justify-between px-10 py-6 bg-white shadow-md z-20">
    <div class="absolute inset-0 bg-[url('/api/placeholder/100/100')] opacity-5 mix-blend-overlay pointer-events-none"></div>
    <div class="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>

    <div class="flex items-center relative">
      <router-link 
        :to="isAuthenticated ? '/chat' : '/'"
        class="font-extrabold text-[#181C3A] text-2xl tracking-tight hover:text-blue-600 transition-colors duration-200 cursor-pointer"
        style="font-family: var(--font-inter)"
        :title="isAuthenticated ? 'Go to CHAT' : 'Go to HOME'"
      >
        CZSU - Multi-Agent Text-to-SQL
      </router-link>
    </div>

    <nav class="flex items-center space-x-6">
      <router-link
        v-for="item in menuItems"
        :key="item.href"
        :to="item.href"
        :class="`text-base px-3 py-2 font-semibold rounded-lg transition-all duration-200 cursor-pointer ` +
          (isActive(item.href)
            ? 'text-[#181C3A] font-extrabold bg-gray-100 shadow-sm '
            : 'text-[#181C3A]/80 hover:text-gray-400 hover:bg-gray-50 ')"
        style="font-family: var(--font-inter)"
      >
        {{ item.label.charAt(0) + item.label.slice(1).toLowerCase() }}
      </router-link>
      <div class="ml-6">
        <AuthButton :compact="true" />
      </div>
    </nav>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { useAuthStore } from '@/stores/auth';
import AuthButton from '@/components/AuthButton.vue';

const route = useRoute();
const authStore = useAuthStore();

const menuItems = [
  { label: 'HOME', href: '/' },
  { label: 'CHAT', href: '/chat' },
  { label: 'CATALOG', href: '/catalog' },
  { label: 'DATA', href: '/data' },
  { label: 'CONTACTS', href: '/contacts' },
];

const isAuthenticated = computed(() => authStore.isAuthenticated);

// Check if current route is active
const isActive = (href: string): boolean => {
  return route.path === href;
};
</script> 