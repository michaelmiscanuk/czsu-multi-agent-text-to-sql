import { useState, useEffect, useCallback } from 'react';
import { authApiFetch } from './api';
import { getSession } from 'next-auth/react';
import { SentimentRequest, SentimentResponse } from '@/types';
import { useChatCache } from '@/contexts/ChatCacheContext';

interface SentimentState {
  [runId: string]: boolean | null; // true = thumbs up, false = thumbs down, null = no sentiment
}

interface UseSentimentReturn {
  sentiments: SentimentState;
  updateSentiment: (runId: string, sentiment: boolean | null) => Promise<void>;
  loadSentiments: (threadId: string) => Promise<void>;
  getSentimentForRunId: (runId: string) => boolean | null;
}

export function useSentiment(): UseSentimentReturn {
  const [sentiments, setSentiments] = useState<SentimentState>({});
  
  // Use ChatCacheContext for cached sentiment data
  const { getSentimentsForThread, updateCachedSentiment } = useChatCache();

  // Load sentiments from localStorage on component mount
  useEffect(() => {
    loadSentimentsFromStorage();
  }, []);

  const saveSentimentsToStorage = useCallback((sentimentData: SentimentState) => {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        localStorage.setItem('czsu-sentiments', JSON.stringify(sentimentData));
        console.log('[SENTIMENT-HOOK] Saved sentiments to localStorage:', Object.keys(sentimentData).length);
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error saving sentiments to localStorage:', error);
    }
  }, []);

  const loadSentimentsFromStorage = useCallback(() => {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem('czsu-sentiments');
        if (stored) {
          const sentimentData = JSON.parse(stored);
          setSentiments(sentimentData);
          console.log('[SENTIMENT-HOOK] Loaded sentiments from localStorage:', Object.keys(sentimentData).length);
        }
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error loading sentiments from localStorage:', error);
    }
  }, []);

  // OPTIMIZED: Load sentiments from cache instead of making API calls
  const loadSentiments = useCallback(async (threadId: string) => {
    console.log('[SENTIMENT-HOOK] Loading sentiments for thread:', threadId, '(using cached data only)');
    
    try {
      // Get cached sentiments for this thread
      const cachedSentiments = getSentimentsForThread(threadId);
      console.log('[SENTIMENT-HOOK] Found cached sentiments:', Object.keys(cachedSentiments).length);
      
      if (cachedSentiments && Object.keys(cachedSentiments).length > 0) {
        // Convert thread-specific sentiments to global sentiment state
        const updatedSentiments = { ...sentiments };
        
        // Add/update sentiments from cache
        Object.entries(cachedSentiments).forEach(([runId, sentiment]) => {
          updatedSentiments[runId] = sentiment;
        });
        
        setSentiments(updatedSentiments);
        saveSentimentsToStorage(updatedSentiments);
        console.log('[SENTIMENT-HOOK] Updated sentiments with cached data:', Object.keys(cachedSentiments).length, 'entries');
      } else {
        console.log('[SENTIMENT-HOOK] No cached sentiments found for thread:', threadId);
        console.log('[SENTIMENT-HOOK] Bulk loading should have loaded all sentiments - no fallback API call needed');
        
        // REMOVED: No more fallback API calls - bulk loading handles everything
        // The bulk loading via /chat/all-messages-for-all-threads should have loaded all sentiments
        // If there are no cached sentiments, it means this thread has no sentiments
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error loading sentiments:', error);
    }
  }, [getSentimentsForThread, sentiments, saveSentimentsToStorage]);

  // OPTIMIZED: Update sentiment using cache and API
  const updateSentiment = useCallback(async (runId: string, sentiment: boolean | null) => {
    console.log('[SENTIMENT-HOOK] Updating sentiment:', { runId, sentiment });

    // Update local state immediately for responsive UI
    setSentiments(prev => {
      const updated = { ...prev, [runId]: sentiment };
      saveSentimentsToStorage(updated);
      return updated;
    });

    // Update cache context
    // Note: We need to find which thread this runId belongs to
    // For now, we'll update the cache when we know the threadId from MessageArea
    // updateCachedSentiment(threadId, runId, sentiment);

    // Send to server
    try {
      const session = await getSession();
      if (!session?.id_token) {
        console.log('[SENTIMENT-HOOK] No session available for sentiment update');
        return;
      }

      const request: SentimentRequest = {
        run_id: runId,
        sentiment
      };

      await authApiFetch('/sentiment', session.id_token, {
        method: 'POST',
        body: JSON.stringify(request)
      });

      console.log('[SENTIMENT-HOOK] Sentiment successfully sent to server');
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error sending sentiment to server:', error);
      // Note: We keep the local state even if server update fails
      // This provides offline functionality and responsive UI
    }
  }, [saveSentimentsToStorage]);

  // Get sentiment for a specific run_id
  const getSentimentForRunId = useCallback((runId: string): boolean | null => {
    // Return null for invalid/empty run IDs to prevent false sentiment lookups
    if (!runId || runId.trim() === '') {
      console.log('[SENTIMENT-HOOK] getSentimentForRunId: Invalid runId, returning null:', { runId });
      return null;
    }
    
    const sentiment = sentiments[runId] ?? null;
    console.log('[SENTIMENT-HOOK] getSentimentForRunId:', { runId, sentiment });
    return sentiment;
  }, [sentiments]);

  return {
    sentiments,
    updateSentiment,
    loadSentiments,
    getSentimentForRunId
  };
} 