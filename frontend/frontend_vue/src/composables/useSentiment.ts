import { ref, type Ref } from 'vue';
import { useAuthStore } from '@/stores/auth';
import { authApiFetch } from '@/lib/api';

interface SentimentState {
  [runId: string]: boolean | null; // true = thumbs up, false = thumbs down, null = no sentiment
}

interface UseSentimentReturn {
  sentiments: Ref<SentimentState>;
  updateSentiment: (runId: string, sentiment: boolean | null) => Promise<void>;
  loadSentiments: (cachedSentiments: SentimentState) => void;
  getSentimentForRunId: (runId: string) => boolean | null;
}

export function useSentiment(): UseSentimentReturn {
  const sentiments = ref<SentimentState>({});
  const authStore = useAuthStore();
  
  const updateSentiment = async (runId: string, sentiment: boolean | null) => {
    try {
      const token = await authStore.getValidToken();
      if (!token) {
        throw new Error('No authentication token available');
      }
      
      // Update local state immediately for better UX
      if (sentiment === null) {
        delete sentiments.value[runId];
      } else {
        sentiments.value[runId] = sentiment;
      }
      
      // Send to API
      await authApiFetch('/sentiment', token, {
        method: 'POST',
        data: {
          run_id: runId,
          sentiment: sentiment,
        },
      });
      
      console.log('[useSentiment] ✅ Sentiment updated for run:', runId, '- sentiment:', sentiment);
      
    } catch (error) {
      console.error('[useSentiment] ❌ Error updating sentiment:', error);
      throw error;
    }
  };
  
  const loadSentiments = (cachedSentiments: SentimentState) => {
    sentiments.value = { ...sentiments.value, ...cachedSentiments };
    console.log('[useSentiment] ✅ Sentiments loaded from cache:', Object.keys(cachedSentiments).length);
  };
  
  const getSentimentForRunId = (runId: string): boolean | null => {
    return sentiments.value[runId] ?? null;
  };
  
  return {
    sentiments,
    updateSentiment,
    loadSentiments,
    getSentimentForRunId,
  };
} 