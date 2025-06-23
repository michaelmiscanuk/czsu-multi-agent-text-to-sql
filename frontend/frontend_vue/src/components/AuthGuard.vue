<template>
  <div>
    <!-- Loading State -->
    <div v-if="authStore.isLoading" class="flex items-center justify-center min-h-[400px]">
      <div class="text-gray-600">Loading...</div>
    </div>

    <!-- For public routes or authenticated users, show content normally -->
    <div v-else-if="isPublicRoute || authStore.isAuthenticated">
      <slot />
    </div>

    <!-- For protected routes when unauthenticated, show login requirement -->
    <div v-else-if="isProtectedRoute && !authStore.isAuthenticated" class="flex flex-col items-center justify-center min-h-[500px] p-8">
      <div class="max-w-md text-center space-y-6">
        <div class="text-6xl mb-4">🔒</div>
        <h1 class="text-2xl font-bold text-gray-800 mb-2">
          {{ getPageTitle() }} - Login Required
        </h1>
        <p class="text-gray-600 mb-6">
          You need to sign in to access the {{ getPageTitle().toLowerCase() }} page.
        </p>
        <div class="space-y-4">
          <AuthButton :compact="false" />
          <p class="text-sm text-gray-500">
            Sign in to start using our CZSU Multi-agent Text-to-SQL System
          </p>
        </div>
      </div>
    </div>

    <!-- For any other unauthenticated routes, redirect to login -->
    <div v-else class="flex items-center justify-center min-h-[400px]">
      <div class="text-gray-600">Redirecting to login...</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import { useAuthStore } from '@/stores/auth';
import AuthButton from '@/components/AuthButton.vue';

// Public routes that don't require authentication
const PUBLIC_ROUTES = ["/", "/contacts", "/login"];

// Routes that should allow navigation but protect content behind login
const PROTECTED_ROUTES = ["/chat", "/catalog", "/data"];

// Composables
const route = useRoute();
const authStore = useAuthStore();

// Computed
const isPublicRoute = computed(() => PUBLIC_ROUTES.includes(route.path));
const isProtectedRoute = computed(() => PROTECTED_ROUTES.includes(route.path));

// Methods
const getPageTitle = (): string => {
  switch (route.path) {
    case "/chat":
      return "Chat";
    case "/catalog":
      return "Catalog";
    case "/data":
      return "Data";
    default:
      return "This Page";
  }
};
</script> 