// Centralized API configuration and utilities
// This file provides consistent API base URL and common fetch configurations

import { ApiConfig, ApiError } from '@/types';

// API Configuration
export const API_CONFIG: ApiConfig = {
  baseUrl: process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000',
  timeout: 600000, // 600 seconds - increased for LangGraph processing
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
  
  try {
    const response = await fetch(url, {
      ...createFetchOptions(options),
      signal: AbortSignal.timeout(API_CONFIG.timeout),
    });

    if (!response.ok) {
      // Try to get error details from response
      let errorDetails = `HTTP error! status: ${response.status}`;
      try {
        const errorText = await response.text();
        if (errorText) {
          errorDetails += ` - Response: ${errorText}`;
        }
      } catch (e) {
        // If we can't read the response, just use the status
        console.warn('Could not read error response:', e);
      }
      throw new Error(errorDetails);
    }

    return await response.json();
  } catch (error) {
    // Add more context to timeout errors
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new Error(`Request timeout after ${API_CONFIG.timeout}ms - ${endpoint}`);
    }
    throw handleApiError(error, `apiFetch ${endpoint}`);
  }
};

// Authenticated API fetch wrapper
export const authApiFetch = async <T>(
  endpoint: string,
  token: string,
  options: RequestInit = {}
): Promise<T> => {
  return apiFetch<T>(endpoint, createAuthFetchOptions(token, options));
}; 