// Central type definitions for the CZSU Multi-Agent Text-to-SQL application
// This file serves as the single source of truth for all shared interfaces

export interface ChatThreadMeta {
  thread_id: string;
  latest_timestamp: string;
  run_count: number;
  title: string;
  full_prompt: string;
}

export interface ChatMessage {
  id: string;
  threadId: string;
  user: string;
  content: string;
  isUser: boolean;
  createdAt: number;
  error?: string;
  meta?: Record<string, any>;
  queriesAndResults?: [string, string][];
  isLoading?: boolean;
  startedAt?: number;
  isError?: boolean;
}

export interface AnalyzeRequest {
  prompt: string;
  thread_id: string;
}

export interface AnalyzeResponse {
  prompt: string;
  result: string;
  queries_and_results: [string, string][];
  thread_id: string;
  top_selection_codes: string[];
  iteration: number;
  max_iterations: number;
  sql: string | null;
  datasetUrl: string | null;
  run_id: string;
  warning?: string;
}

export interface FeedbackRequest {
  run_id: string;
  feedback: number; // 1 for thumbs up, 0 for thumbs down
  comment?: string;
}

export interface ChatThreadResponse {
  thread_id: string;
  latest_timestamp: string;
  run_count: number;
  title: string;
  full_prompt: string;
}

export interface ApiError {
  detail: string;
  status?: number;
}

// Configuration types
export interface ApiConfig {
  baseUrl: string;
  timeout: number;
} 