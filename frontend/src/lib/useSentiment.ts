import { useState, useEffect, useCallback } from 'react';
import { authApiFetch } from './api';
import { getSession } from 'next-auth/react';
import { SentimentRequest, SentimentResponse } from '@/types';

interface SentimentState {
  [runId: string]: boolean | null; // true = thumbs up, false = thumbs down, null = no sentiment
}

interface UseSentimentReturn {
  sentiments: SentimentState;
  updateSentiment: (runId: string, sentiment: boolean | null) => Promise<void>;
  loadSentiments: (threadId: string) => Promise<void>;
  getSentimentForRunId: (runId: string) => boolean | null;
}

const SENTIMENT_STORAGE_KEY = 'czsu-sentiment-data';

export function useSentiment(): UseSentimentReturn {
  const [sentiments, setSentiments] = useState<SentimentState>({});

  // Load sentiments from localStorage on mount
  useEffect(() => {
    try {
      const savedSentiments = localStorage.getItem(SENTIMENT_STORAGE_KEY);
      if (savedSentiments) {
        const parsedSentiments = JSON.parse(savedSentiments);
        setSentiments(parsedSentiments);
        console.log('[SENTIMENT-HOOK] Loaded sentiments from localStorage:', parsedSentiments);
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error loading sentiments from localStorage:', error);
    }
  }, []);

  // Save sentiments to localStorage whenever state changes
  const saveSentimentsToStorage = useCallback((newSentiments: SentimentState) => {
    try {
      localStorage.setItem(SENTIMENT_STORAGE_KEY, JSON.stringify(newSentiments));
      console.log('[SENTIMENT-HOOK] Saved sentiments to localStorage:', newSentiments);
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error saving sentiments to localStorage:', error);
    }
  }, []);

  // Load sentiments from the server for a specific thread
  const loadSentiments = useCallback(async (threadId: string) => {
    try {
      console.log('[SENTIMENT-HOOK] Loading sentiments for thread:', threadId);
      const session = await getSession();
      if (!session?.id_token) {
        console.log('[SENTIMENT-HOOK] No session available for loading sentiments');
        return;
      }

      const response = await authApiFetch<SentimentResponse>(
        `/chat/${threadId}/sentiments`,
        session.id_token
      );

      console.log('[SENTIMENT-HOOK] Received sentiments from server:', response.sentiments);
      
      // Use server sentiments as the source of truth, overriding localStorage
      if (response.sentiments && Object.keys(response.sentiments).length > 0) {
        setSentiments(response.sentiments);
        saveSentimentsToStorage(response.sentiments);
        console.log('[SENTIMENT-HOOK] Updated sentiments with server data:', response.sentiments);
      } else {
        console.log('[SENTIMENT-HOOK] No sentiments found on server for thread:', threadId);
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error loading sentiments from server:', error);
    }
  }, [saveSentimentsToStorage]);

  // Update sentiment (both locally and on server)
  const updateSentiment = useCallback(async (runId: string, sentiment: boolean | null) => {
    console.log('[SENTIMENT-HOOK] Updating sentiment:', { runId, sentiment });

    // Update local state immediately for responsive UI
    setSentiments(prev => {
      const updated = { ...prev, [runId]: sentiment };
      saveSentimentsToStorage(updated);
      return updated;
    });

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
    const sentiment = sentiments[runId] ?? null;
    console.log('[SENTIMENT-HOOK] getSentimentForRunId:', { runId, sentiment, allSentiments: sentiments });
    return sentiment;
  }, [sentiments]);

  return {
    sentiments,
    updateSentiment,
    loadSentiments,
    getSentimentForRunId
  };
} 