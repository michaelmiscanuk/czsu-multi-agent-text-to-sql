// Centralized API configuration and utilities
// This file provides consistent API base URL and common fetch configurations

import { ApiConfig, ApiError } from '@/types';
import { getSession } from 'next-auth/react';

// API Configuration
export const API_CONFIG: ApiConfig = {
  baseUrl: process.env.NEXT_PUBLIC_API_BASE || 'https://czsu-multi-agent-text-to-sql.onrender.com',
  timeout: 600000, // 10 minutes - increased to match backend analysis timeout
};

// Common fetch options factory
export const createFetchOptions = (options: RequestInit = {}): RequestInit => {
  return {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };
};

// Auth fetch options factory
export const createAuthFetchOptions = (token: string, options: RequestInit = {}): RequestInit => {
  return {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...options.headers,
    },
  };
};

// API Error handler
export const handleApiError = (error: any, context: string): ApiError => {
  console.error(`[API-Error] ${context}:`, error);
  
  if (error instanceof Error) {
    return {
      detail: error.message,
      status: (error as any).status || 500,
    };
  }
  
  return {
    detail: 'An unexpected error occurred',
    status: 500,
  };
};

// API fetch wrapper with error handling
export const apiFetch = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => {
  const url = `${API_CONFIG.baseUrl}${endpoint}`;
  
  console.log(`[API-Fetch] 🚀 Starting request to: ${endpoint}`);
  console.log(`[API-Fetch] 📋 Request options:`, {
    method: options.method || 'GET',
    headers: options.headers,
    bodyLength: options.body ? (options.body as string).length : 0,
    timeout: API_CONFIG.timeout
  });
  
  try {
    const startTime = Date.now();
    
    const response = await fetch(url, {
      ...createFetchOptions(options),
      signal: AbortSignal.timeout(API_CONFIG.timeout),
    });

    const responseTime = Date.now() - startTime;
    console.log(`[API-Fetch] ⏱️ Response time: ${responseTime}ms`);
    console.log(`[API-Fetch] 📊 Response status: ${response.status} ${response.statusText}`);
    console.log(`[API-Fetch] 📋 Response headers:`, Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      console.error(`[API-Fetch] ❌ HTTP error! status: ${response.status}`);
      // Try to get error details from response
      let errorDetails = `HTTP error! status: ${response.status}`;
      try {
        const errorText = await response.text();
        console.error(`[API-Fetch] ❌ Error response body:`, errorText);
        if (errorText) {
          errorDetails += ` - Response: ${errorText}`;
        }
      } catch (e) {
        // If we can't read the response, just use the status
        console.warn('[API-Fetch] ⚠️ Could not read error response:', e);
      }
      
      // Create error with status code for 401 handling
      const error = new Error(errorDetails);
      (error as any).status = response.status;
      throw error;
    }

    console.log(`[API-Fetch] 📖 Attempting to parse JSON response...`);
    const jsonData = await response.json();
    console.log(`[API-Fetch] ✅ JSON parsed successfully, data size:`, JSON.stringify(jsonData).length, 'characters');
    console.log(`[API-Fetch] 📋 Response data preview:`, {
      ...jsonData,
      result: jsonData.result ? `${String(jsonData.result).substring(0, 100)}...` : undefined
    });
    
    return jsonData;
  } catch (error) {
    console.error(`[API-Fetch] ❌ Request failed:`, error);
    console.error(`[API-Fetch] ❌ Error type:`, error instanceof Error ? error.constructor.name : typeof error);
    console.error(`[API-Fetch] ❌ Error message:`, error instanceof Error ? error.message : String(error));
    
    // Add more context to timeout errors
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      const timeoutError = new Error(`Request timeout after ${API_CONFIG.timeout}ms - ${endpoint}`);
      console.error(`[API-Fetch] ⏰ Timeout error created:`, timeoutError.message);
      throw timeoutError;
    }
    throw error; // Preserve original error with status for 401 handling
  }
};

// Enhanced authenticated API fetch wrapper with automatic token refresh
export const authApiFetch = async <T>(
  endpoint: string,
  token: string,
  options: RequestInit = {}
): Promise<T> => {
  console.log(`[AuthAPI-Fetch] 🔐 Starting authenticated request to: ${endpoint}`);
  
  try {
    // First attempt with provided token
    return await apiFetch<T>(endpoint, createAuthFetchOptions(token, options));
  } catch (error: any) {
    // Check if this is a 401 Unauthorized error (token expired)
    if (error && error.status === 401) {
      console.log(`[AuthAPI-Fetch] 🔄 Received 401 Unauthorized - attempting token refresh...`);
      
      try {
        // Attempt to refresh the session
        console.log(`[AuthAPI-Fetch] 🔄 Calling getSession() to refresh token...`);
        const freshSession = await getSession();
        
        if (freshSession && freshSession.id_token) {
          console.log(`[AuthAPI-Fetch] ✅ Got fresh token - retrying original request...`);
          
          // Retry the original request with the fresh token
          return await apiFetch<T>(endpoint, createAuthFetchOptions(freshSession.id_token, options));
        } else {
          console.error(`[AuthAPI-Fetch] ❌ No fresh session available after refresh attempt`);
          console.error(`[AuthAPI-Fetch] 📋 Session state:`, {
            hasSession: !!freshSession,
            hasIdToken: !!(freshSession && freshSession.id_token),
            sessionKeys: freshSession ? Object.keys(freshSession) : []
          });
          
          // If we can't get a fresh token, throw an authentication error
          throw new Error('Authentication failed - please log in again');
        }
      } catch (refreshError) {
        console.error(`[AuthAPI-Fetch] ❌ Token refresh failed:`, refreshError);
        
        // If token refresh fails, throw a user-friendly error
        throw new Error('Session expired - please refresh the page and log in again');
      }
    } else {
      // For non-401 errors, just re-throw the original error
      console.log(`[AuthAPI-Fetch] ❌ Non-401 error, re-throwing:`, error);
      throw error;
    }
  }
}; 