<template>
  <div class="min-h-screen w-full bg-gradient-to-br from-blue-100 via-blue-50 to-blue-200 flex flex-col">
    <!-- Header -->
    <div class="sticky top-0 z-50">
      <Header />
    </div>
    
    <!-- Main Content -->
    <main class="main-container-unified">
      <AuthGuard>
        <router-view v-slot="{ Component }">
          <transition name="page" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </AuthGuard>
    </main>
    
    <!-- Footer (conditional) -->
    <footer 
      v-if="!shouldHideFooter" 
      class="w-full text-center text-gray-400 text-sm py-4 mt-4"
    >
      &copy; {{ new Date().getFullYear() }} Michael Miscanuk. Data from the Czech Statistical Office (CZSU).
    </footer>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import Header from '@/components/Header.vue';
import AuthGuard from '@/components/AuthGuard.vue';

const route = useRoute();

// Determine if footer should be hidden based on current route
const shouldHideFooter = computed(() => {
  const pathname = route.path;
  const isChatPage = pathname === '/chat';
  const isCatalogPage = pathname === '/catalog';
  const isDataPage = pathname === '/data';
  
  return isChatPage || isCatalogPage || isDataPage;
});
</script>

<style scoped>
/* Page transition animations */
.page-enter-active,
.page-leave-active {
  transition: opacity 0.2s ease;
}

.page-enter-from,
.page-leave-to {
  opacity: 0;
}
</style> 