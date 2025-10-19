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
  createdAt: number;
  prompt?: string;
  final_answer?: string;
  followup_prompts?: string[]; // Add follow-up prompt suggestions
  queries_and_results?: [string, string][];
  datasets_used?: string[];
  top_chunks?: Array<{ 
    page_content: string; 
    content?: string; // fallback for existing usage
    source_file?: string;
    page_number?: number;
    metadata?: Record<string, any>;
  }>;
  sql_query?: string;
  error?: string;
  isLoading?: boolean;
  startedAt?: number;
  isError?: boolean;
  run_id?: string; // Add run_id to store the LangSmith run identifier
}

export interface AnalyzeRequest {
  prompt: string;
  thread_id: string;
  run_id?: string; // Optional run_id for tracking execution
}

export interface AnalyzeResponse {
  prompt: string;
  result: string;
  followup_prompts?: string[]; // Add follow-up prompt suggestions
  queries_and_results: [string, string][];
  thread_id: string;
  top_selection_codes: string[];
  datasets_used?: string[]; // Add this field as optional since backend might send it
  iteration: number;
  max_iterations: number;
  sql: string | null;
  datasetUrl: string | null;
  run_id: string;
  warning?: string;
  top_chunks?: Array<{
    page_content: string;
    content?: string; // fallback for existing usage
    source_file?: string;
    page_number?: number;
    metadata?: Record<string, any>;
  }>;
}

export interface FeedbackRequest {
  run_id: string;
  feedback: number; // 1 for thumbs up, 0 for thumbs down
  comment?: string;
}

export interface SentimentRequest {
  run_id: string;
  sentiment: boolean | null; // true for thumbs up, false for thumbs down, null to clear
}

export interface SentimentResponse {
  sentiments: { [run_id: string]: boolean | null };
}

export interface ChatThreadResponse {
  thread_id: string;
  latest_timestamp: string;
  run_count: number;
  title: string;
  full_prompt: string;
}

export interface PaginatedChatThreadsResponse {
  threads: ChatThreadResponse[];
  total_count: number;
  page: number;
  limit: number;
  has_more: boolean;
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