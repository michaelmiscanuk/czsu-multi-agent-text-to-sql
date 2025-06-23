import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

export interface User {
  id: string;
  name?: string;
  email?: string;
  image?: string;
}

export interface Session {
  user: User;
  accessToken: string;
  idToken?: string;
  refreshToken?: string;
  expiresAt: number;
}

export const useAuthStore = defineStore('auth', () => {
  // State
  const session = ref<Session | null>(null);
  const isLoading = ref(false);
  const error = ref<string | null>(null);
  
  // Computed
  const isAuthenticated = computed(() => !!session.value);
  const user = computed(() => session.value?.user || null);
  const userEmail = computed(() => session.value?.user?.email || null);
  
  // Auth token management
  const getValidToken = async (): Promise<string | null> => {
    if (!session.value) return null;
    
    // Check if token is expired
    if (Date.now() >= session.value.expiresAt) {
      console.log('[AuthStore] Token expired, attempting refresh');
      await refreshSession();
    }
    
    return session.value?.idToken || session.value?.accessToken || null;
  };
  
  // Initialize Google OAuth
  const initializeGoogleAuth = async () => {
    try {
      // This will be implemented based on the Google OAuth library chosen
      console.log('[AuthStore] Initializing Google OAuth');
      
      // For now, we'll use a placeholder implementation
      // In a real implementation, you would initialize the Google OAuth library here
      
    } catch (error) {
      console.error('[AuthStore] Failed to initialize Google OAuth:', error);
    }
  };
  
  // Sign in with Google
  const signInWithGoogle = async () => {
    isLoading.value = true;
    error.value = null;
    
    try {
      // This would integrate with Google OAuth library
      // For now, it's a placeholder implementation
      
      // Example flow:
      // 1. Trigger Google OAuth popup/redirect
      // 2. Get authorization code
      // 3. Exchange for tokens
      // 4. Get user info
      // 5. Store session
      
      console.log('[AuthStore] Signing in with Google');
      
      // Placeholder - replace with actual Google OAuth implementation
      throw new Error('Google OAuth not yet implemented');
      
    } catch (err) {
      console.error('[AuthStore] Sign in failed:', err);
      error.value = err instanceof Error ? err.message : 'Sign in failed';
    } finally {
      isLoading.value = false;
    }
  };
  
  // Refresh session
  const refreshSession = async () => {
    if (!session.value?.refreshToken) {
      console.log('[AuthStore] No refresh token available');
      await signOut();
      return;
    }
    
    try {
      console.log('[AuthStore] Refreshing session');
      
      // This would call your refresh token endpoint
      // For now, it's a placeholder
      
      // Example:
      // const response = await authApiFetch('/auth/refresh', session.value.refreshToken);
      // updateSession(response);
      
      console.log('[AuthStore] Session refresh not yet implemented');
      
    } catch (error) {
      console.error('[AuthStore] Session refresh failed:', error);
      await signOut();
    }
  };
  
  // Sign out
  const signOut = async () => {
    try {
      console.log('[AuthStore] Signing out');
      
      // Clear session
      session.value = null;
      
      // Clear localStorage
      localStorage.removeItem('auth-session');
      localStorage.removeItem('auth-token');
      
      // Clear any other auth-related storage
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith('czsu-') || key.startsWith('auth-')) {
          localStorage.removeItem(key);
        }
      });
      
      console.log('[AuthStore] Signed out successfully');
      
    } catch (error) {
      console.error('[AuthStore] Sign out error:', error);
    }
  };
  
  // Update session
  const updateSession = (newSession: Session) => {
    session.value = newSession;
    
    // Persist to localStorage
    try {
      localStorage.setItem('auth-session', JSON.stringify(newSession));
      localStorage.setItem('auth-token', newSession.idToken || newSession.accessToken);
    } catch (error) {
      console.error('[AuthStore] Failed to persist session:', error);
    }
  };
  
  // Load session from storage
  const loadStoredSession = () => {
    try {
      const stored = localStorage.getItem('auth-session');
      if (!stored) return false;
      
      const parsedSession: Session = JSON.parse(stored);
      
      // Check if session is expired
      if (Date.now() >= parsedSession.expiresAt) {
        console.log('[AuthStore] Stored session expired');
        localStorage.removeItem('auth-session');
        localStorage.removeItem('auth-token');
        return false;
      }
      
      session.value = parsedSession;
      console.log('[AuthStore] Loaded session from storage for user:', parsedSession.user.email);
      return true;
      
    } catch (error) {
      console.error('[AuthStore] Failed to load stored session:', error);
      localStorage.removeItem('auth-session');
      localStorage.removeItem('auth-token');
      return false;
    }
  };
  
  // Initialize auth (call this on app startup)
  const initialize = async () => {
    console.log('[AuthStore] Initializing authentication');
    
    // Try to load existing session
    const hasStoredSession = loadStoredSession();
    
    if (hasStoredSession) {
      console.log('[AuthStore] Found valid stored session');
    } else {
      console.log('[AuthStore] No valid stored session found');
    }
    
    // Initialize Google OAuth
    await initializeGoogleAuth();
  };
  
  // Handle OAuth callback (for redirect flow)
  const handleOAuthCallback = async (code: string, state?: string) => {
    isLoading.value = true;
    error.value = null;
    
    try {
      console.log('[AuthStore] Handling OAuth callback');
      
      // Exchange authorization code for tokens
      // This would call your backend OAuth endpoint
      
      // Placeholder implementation
      throw new Error('OAuth callback handling not yet implemented');
      
    } catch (err) {
      console.error('[AuthStore] OAuth callback failed:', err);
      error.value = err instanceof Error ? err.message : 'OAuth callback failed';
    } finally {
      isLoading.value = false;
    }
  };
  
  return {
    // State
    session: computed(() => session.value),
    isLoading: computed(() => isLoading.value),
    error: computed(() => error.value),
    isAuthenticated,
    user,
    userEmail,
    
    // Actions
    signInWithGoogle,
    signOut,
    refreshSession,
    getValidToken,
    initialize,
    handleOAuthCallback,
    updateSession,
  };
}); 