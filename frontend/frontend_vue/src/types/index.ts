// Shared types across the application

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
  top_chunks?: Array<{
    content: string;
    metadata: Record<string, any>;
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

export interface ApiConfig {
  baseUrl: string;
  timeout: number;
} 