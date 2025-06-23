import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';
import type { ApiError, ApiConfig } from '@/types';

// API Configuration
export const API_CONFIG: ApiConfig = {
  baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 300000, // 5 minutes
};

// For convenience, also export BASE_URL for backward compatibility
export const BASE_URL = API_CONFIG.baseUrl;

// Create axios instance
const apiClient = axios.create({
  baseURL: API_CONFIG.baseUrl,
  timeout: API_CONFIG.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API] ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('[API] Response error:', error);
    return Promise.reject(error);
  }
);

export const createFetchOptions = (options: AxiosRequestConfig = {}): AxiosRequestConfig => {
  return {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };
};

export const createAuthFetchOptions = (token: string, options: AxiosRequestConfig = {}): AxiosRequestConfig => {
  return {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...options.headers,
    },
    ...options,
  };
};

export const handleApiError = (error: any, context: string): ApiError => {
  console.error(`[API Error - ${context}]`, error);
  
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    return {
      detail: data?.detail || data?.message || `HTTP ${status} Error`,
      status: status,
    };
  } else if (error.request) {
    // Network error
    return {
      detail: 'Network error - please check your connection',
      status: 0,
    };
  } else {
    // Other error
    return {
      detail: error.message || 'An unexpected error occurred',
      status: undefined,
    };
  }
};

export const apiFetch = async <T>(
  endpoint: string,
  options: AxiosRequestConfig = {}
): Promise<T> => {
  try {
    const config = createFetchOptions(options);
    const response: AxiosResponse<T> = await apiClient.request({
      url: endpoint,
      ...config,
    });
    
    return response.data;
  } catch (error: any) {
    const apiError = handleApiError(error, `apiFetch: ${endpoint}`);
    
    // Re-throw as a more specific error type
    const enhancedError = new Error(apiError.detail);
    (enhancedError as any).status = apiError.status;
    (enhancedError as any).isApiError = true;
    
    throw enhancedError;
  }
};

export const authApiFetch = async <T>(
  endpoint: string,
  token: string,
  options: AxiosRequestConfig = {}
): Promise<T> => {
  try {
    const config = createAuthFetchOptions(token, options);
    const response: AxiosResponse<T> = await apiClient.request({
      url: endpoint,
      ...config,
    });
    
    return response.data;
  } catch (error: any) {
    const apiError = handleApiError(error, `authApiFetch: ${endpoint}`);
    
    // Re-throw as a more specific error type
    const enhancedError = new Error(apiError.detail);
    (enhancedError as any).status = apiError.status;
    (enhancedError as any).isApiError = true;
    
    throw enhancedError;
  }
}; 