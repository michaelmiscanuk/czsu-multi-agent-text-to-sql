<template>
  <div>
    <span v-if="authStore.isLoading" class="text-xs text-gray-400">Loading...</span>
    
    <div v-else-if="authStore.isAuthenticated" class="flex items-center space-x-2">
      <img
        v-if="user?.image"
        :src="user.image"
        :alt="user?.name || user?.email || 'avatar'"
        class="w-7 h-7 rounded-full border border-gray-300 bg-gray-100 object-cover"
        @error="handleImageError"
      />
      <div
        v-else
        class="w-7 h-7 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold text-gray-700"
      >
        {{ getUserInitials(user?.name || user?.email || '?') }}
      </div>
      
      <span class="text-xs text-gray-700 font-medium">{{ user?.name || user?.email }}</span>
      
      <button
        class="px-3 py-1.5 text-xs bg-gradient-to-r from-gray-200 to-gray-300 hover:from-gray-300 hover:to-gray-400 rounded-lg font-semibold text-gray-700 border border-gray-300 transition-all duration-200 shadow-sm"
        @click="handleSignOut"
      >
        Sign out
      </button>
    </div>
    
    <div v-else>
      <!-- Compact version for header -->
      <button
        v-if="compact"
        class="flex items-center px-7 py-2 bg-white rounded-full border border-gray-200 shadow-md text-[#172153] font-bold hover:bg-gray-200 hover:shadow-lg hover:border-gray-300 transition-all duration-300 focus:outline-none"
        @click="handleSignIn"
        style="min-width: 110px"
      >
        Log In
      </button>
      
      <!-- Main Google button -->
      <button
        v-else
        class="flex items-center justify-center w-full px-4 py-2 border border-gray-300 rounded-lg shadow-sm bg-white hover:bg-blue-50 hover:shadow-lg hover:border-blue-400 transition-all duration-200 text-gray-700 font-medium text-base focus:outline-none"
        @click="handleSignIn"
        style="min-width: 220px"
      >
        <img
          src="https://developers.google.com/identity/images/g-logo.png"
          alt="Google logo"
          class="w-5 h-5 mr-3"
        />
        Continue with Google
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useAuthStore } from '@/stores/auth';
import { useChatCacheStore } from '@/stores/chatCache';

// Props
interface Props {
  compact?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  compact: false,
});

// Composables
const authStore = useAuthStore();
const chatCacheStore = useChatCacheStore();
const router = useRouter();

// Computed
const user = computed(() => authStore.user);

// Methods
const getUserInitials = (name: string): string => {
  return name
    .split(' ')
    .map(s => s[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();
};

const handleImageError = (e: Event) => {
  const target = e.target as HTMLImageElement;
  target.style.display = 'none';
  
  // Create fallback element
  const fallback = document.createElement('div');
  fallback.className = 'w-7 h-7 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold text-gray-700';
  fallback.innerText = getUserInitials(user.value?.name || user.value?.email || '?');
  target.parentNode?.insertBefore(fallback, target.nextSibling);
};

const handleSignIn = async () => {
  try {
    await authStore.signInWithGoogle();
  } catch (error) {
    console.error('[AuthButton] Sign in failed:', error);
  }
};

const handleSignOut = async () => {
  try {
    console.log('[AuthButton] 🚪 Signing out - clearing localStorage for clean state');
    
    // Use the comprehensive cache clearing function for user changes
    chatCacheStore.clearCacheForUserChange(null); // null = no new user (logout)
    
    console.log('[AuthButton] ✅ Cache cleanup completed - next user will have clean state');
    
    // Now perform the actual sign out
    await authStore.signOut();
    
  } catch (error) {
    console.error('[AuthButton] ❌ Error during sign out cleanup:', error);
    // Still perform sign out even if cleanup fails
    await authStore.signOut();
  }
};
</script> 