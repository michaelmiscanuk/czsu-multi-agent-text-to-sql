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

// Mount app
app.mount('#app');

console.log('[Main] 🚀 Vue application started successfully'); 