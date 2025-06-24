import { defineStore } from 'pinia';
import { ref, computed, readonly } from 'vue';

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
      console.log('[AuthStore] Initializing Google OAuth');
      
      // Load Google OAuth script if not already loaded
      if (!window.google) {
        const script = document.createElement('script');
        script.src = 'https://accounts.google.com/gsi/client';
        script.async = true;
        script.defer = true;
        document.head.appendChild(script);
        
        await new Promise((resolve) => {
          script.onload = resolve;
        });
      }
      
      // Initialize Google OAuth
      if (window.google) {
        window.google.accounts.id.initialize({
          client_id: import.meta.env.VITE_GOOGLE_CLIENT_ID || 'your_google_client_id_here',
          callback: handleGoogleCallback,
        });
      }
      
    } catch (error) {
      console.error('[AuthStore] Failed to initialize Google OAuth:', error);
    }
  };
  
  // Handle Google OAuth callback
  const handleGoogleCallback = async (response: any) => {
    try {
      console.log('[AuthStore] Google OAuth callback received');
      
      // Decode the JWT token to get user info
      const payload = JSON.parse(atob(response.credential.split('.')[1]));
      
      const newSession: Session = {
        user: {
          id: payload.sub,
          name: payload.name,
          email: payload.email,
          image: payload.picture,
        },
        accessToken: response.credential,
        idToken: response.credential,
        expiresAt: payload.exp * 1000, // Convert to milliseconds
      };
      
      updateSession(newSession);
      console.log('[AuthStore] Successfully signed in with Google');
      
    } catch (err) {
      console.error('[AuthStore] Failed to handle Google callback:', err);
      error.value = err instanceof Error ? err.message : 'Failed to sign in with Google';
    }
  };
  
  // Sign in with Google
  const signInWithGoogle = async () => {
    isLoading.value = true;
    error.value = null;
    
    try {
      console.log('[AuthStore] Signing in with Google');
      
      if (window.google) {
        window.google.accounts.id.prompt();
      } else {
        // Fallback: create a temporary sign-in for development
        console.log('[AuthStore] Google OAuth not available, using development mode');
        
        const devSession: Session = {
          user: {
            id: 'dev-user',
            name: 'Development User',
            email: 'dev@example.com',
            image: 'https://via.placeholder.com/40',
          },
          accessToken: 'dev-token',
          idToken: 'dev-token',
          expiresAt: Date.now() + 24 * 60 * 60 * 1000, // 24 hours
        };
        
        updateSession(devSession);
        console.log('[AuthStore] Signed in with development account');
      }
      
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
      
      // For Google OAuth, we typically can't refresh tokens client-side
      // In a real app, you'd send the refresh token to your backend
      // For now, we'll just sign out when the token expires
      
      console.log('[AuthStore] Session refresh not implemented for Google OAuth');
      await signOut();
      
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
      
      // Sign out from Google if available
      if (window.google?.accounts?.id) {
        window.google.accounts.id.disableAutoSelect();
      }
      
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
      
      // This would typically involve exchanging the code for tokens
      // with your backend server
      
      // For now, we'll just log that the callback was received
      console.log('[AuthStore] OAuth callback code:', code);
      
    } catch (err) {
      console.error('[AuthStore] OAuth callback failed:', err);
      error.value = err instanceof Error ? err.message : 'OAuth callback failed';
    } finally {
      isLoading.value = false;
    }
  };
  
  return {
    // State
    session: readonly(session),
    isLoading: readonly(isLoading),
    error: readonly(error),
    
    // Computed
    isAuthenticated,
    user,
    userEmail,
    
    // Actions
    getValidToken,
    signInWithGoogle,
    signOut,
    initialize,
    handleOAuthCallback,
  };
});

// Declare global Google OAuth types
declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: any) => void;
          prompt: () => void;
          disableAutoSelect: () => void;
        };
      };
    };
  }
} 