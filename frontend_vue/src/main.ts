import { createApp } from 'vue';
import { createPinia } from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';
import router from './router';
import App from './App.vue';

// Import styles
import './style/main.css';

// Create Vue app
const app = createApp(App);

// Create Pinia store
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate);

// Use plugins
app.use(pinia);
app.use(router);

// Initialize auth store after Pinia is set up
import { useAuthStore } from './stores/auth';

// Mount app
app.mount('#app');

// Initialize auth after mounting
setTimeout(() => {
  const authStore = useAuthStore();
  authStore.initialize();
}, 100);

console.log('[Main] 🚀 Vue application started successfully'); 