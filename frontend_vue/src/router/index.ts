import { createRouter, createWebHistory } from 'vue-router';
import { useAuthStore } from '@/stores/auth';

// Lazy load components for better performance
const Home = () => import('@/views/Home.vue');
const Login = () => import('@/views/Login.vue');
const Chat = () => import('@/views/Chat.vue');
const Catalog = () => import('@/views/Catalog.vue');
const Data = () => import('@/views/Data.vue');
const Contacts = () => import('@/views/Contacts.vue');

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
    meta: { requiresAuth: false },
  },
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { requiresAuth: false },
  },
  {
    path: '/chat',
    name: 'Chat',
    component: Chat,
    meta: { requiresAuth: true },
  },
  {
    path: '/catalog',
    name: 'Catalog',
    component: Catalog,
    meta: { requiresAuth: true },
  },
  {
    path: '/data',
    name: 'Data',
    component: Data,
    meta: { requiresAuth: true },
  },
  {
    path: '/contacts',
    name: 'Contacts',
    component: Contacts,
    meta: { requiresAuth: false },
  },
  // Catch-all route for 404s
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue'),
    meta: { requiresAuth: false },
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

// Navigation guards
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore();
  
  console.log('[Router] Navigating to:', to.path, '- requires auth:', to.meta.requiresAuth);
  
  // Initialize auth store if not already done
  if (!authStore.session && !authStore.isLoading) {
    await authStore.initialize();
  }
  
  // Check if route requires authentication
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    console.log('[Router] Route requires auth, redirecting to login');
    next({ name: 'Login', query: { redirect: to.fullPath } });
    return;
  }
  
  // Redirect from login if already authenticated
  if (to.name === 'Login' && authStore.isAuthenticated) {
    console.log('[Router] Already authenticated, redirecting from login');
    const redirect = to.query.redirect as string;
    next(redirect || { name: 'Chat' });
    return;
  }
  
  next();
});

// After navigation
router.afterEach((to, from) => {
  console.log('[Router] Navigation completed:', from.path, '->', to.path);
  
  // Update page title
  const baseTitle = 'CZSU Multi-Agent Text-to-SQL';
  const pageTitle = to.name === 'Home' ? baseTitle : `${to.name} | ${baseTitle}`;
  document.title = pageTitle;
});

export default router; 