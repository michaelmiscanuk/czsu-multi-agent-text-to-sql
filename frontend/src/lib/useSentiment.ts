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
  // CRITICAL CHANGE: Remove localStorage completely - always use database
  const [sentiments, setSentiments] = useState<SentimentState>({});
  
  // Use ChatCacheContext ONLY for database-loaded sentiment data
  const { getSentimentsForThread, updateCachedSentiment } = useChatCache();

  // CRITICAL: Clear any old localStorage sentiment data on mount
  useEffect(() => {
    try {
      if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem('czsu-sentiments');
        if (stored) {
          localStorage.removeItem('czsu-sentiments');
          console.log('[SENTIMENT-HOOK] CLEARED old localStorage sentiment data - now using DATABASE ONLY');
        }
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error clearing old localStorage data:', error);
    }
  }, []);

  // CRITICAL: Load sentiments ONLY from database via bulk loading
  const loadSentiments = useCallback(async (threadId: string) => {
    console.log('[SENTIMENT-HOOK] Loading sentiments for thread:', threadId, '(DATABASE ONLY - no localStorage)');
    
    try {
      // Get database-loaded sentiments for this thread from bulk loading
      const databaseSentiments = getSentimentsForThread(threadId);
      console.log('[SENTIMENT-HOOK] Found database sentiments:', Object.keys(databaseSentiments).length);
      
      if (databaseSentiments && Object.keys(databaseSentiments).length > 0) {
        // Update local state with database sentiments only
        setSentiments(prev => {
          const updated = { ...prev };
          
          // Clear any previous sentiments for this thread to prevent contamination
          // Only keep sentiments from database
          Object.keys(prev).forEach(runId => {
            if (!Object.keys(databaseSentiments).includes(runId)) {
              delete updated[runId];
            }
          });
          
          // Add database sentiments
          Object.entries(databaseSentiments).forEach(([runId, sentiment]) => {
            updated[runId] = sentiment;
          });
          
          return updated;
        });
        
        console.log('[SENTIMENT-HOOK] Updated sentiments with DATABASE data only:', Object.keys(databaseSentiments).length, 'entries');
      } else {
        console.log('[SENTIMENT-HOOK] No database sentiments found for thread:', threadId);
        
        // Clear any cached sentiments since database has none
        setSentiments(prev => {
          const updated = { ...prev };
          // Remove all sentiments - if not in database, shouldn't be shown
          return {};
        });
      }
    } catch (error) {
      console.error('[SENTIMENT-HOOK] Error loading sentiments from database:', error);
    }
  }, [getSentimentsForThread]);

  // CRITICAL: Update sentiment with optimistic UI + retry logic - always responsive, graceful failure
  const updateSentiment = useCallback(async (runId: string, sentiment: boolean | null) => {
    console.log('[SENTIMENT-HOOK] Updating sentiment (OPTIMISTIC UI + RETRY):', { runId, sentiment });

    // Validate runId before proceeding
    if (!runId) {
      console.log('[SENTIMENT-HOOK] ⚠️ No runId provided - skipping update');
      return;
    }

    // 🚀 STEP 1: Update UI IMMEDIATELY for responsive user experience
    setSentiments(prev => {
      const updated = { ...prev, [runId]: sentiment };
      console.log('[SENTIMENT-HOOK] ✅ UI updated IMMEDIATELY for responsive UX');
      return updated;
    });

    // 🔄 STEP 2: Update database in BACKGROUND with retry logic
    const maxRetries = 3;
    const retryDelay = 1000; // 1 second between retries
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const session = await getSession();
        if (!session?.id_token) {
          console.log(`[SENTIMENT-HOOK] ⚠️ No session available (attempt ${attempt}/${maxRetries})`);
          if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            continue;
          }
          // If no session after all retries, silently revert
          console.log('[SENTIMENT-HOOK] 🔄 No session after retries - silently reverting UI');
          setSentiments(prev => {
            const reverted = { ...prev };
            delete reverted[runId];
            return reverted;
          });
          return;
        }

        const request: SentimentRequest = {
          run_id: runId,
          sentiment
        };

        // Database update happens in background
        await authApiFetch('/sentiment', session.id_token, {
          method: 'POST',
          body: JSON.stringify(request)
        });

        console.log(`[SENTIMENT-HOOK] ✅ Database successfully updated (attempt ${attempt}/${maxRetries})`);
        return; // Success! Exit retry loop
        
      } catch (error) {
        console.error(`[SENTIMENT-HOOK] ❌ Database update failed (attempt ${attempt}/${maxRetries}):`, error);
        
        if (attempt < maxRetries) {
          // Wait before retrying
          console.log(`[SENTIMENT-HOOK] 🔄 Retrying in ${retryDelay}ms...`);
          await new Promise(resolve => setTimeout(resolve, retryDelay));
        } else {
          // All retries failed - silently revert UI state
          console.log('[SENTIMENT-HOOK] 🔄 All retries failed - silently reverting UI state');
          setSentiments(prev => {
            const reverted = { ...prev };
            delete reverted[runId]; // Remove the failed update
            return reverted;
          });
        }
      }
    }
  }, []);

  // CRITICAL: Get sentiment - only show if it exists in database
  const getSentimentForRunId = useCallback((runId: string): boolean | null => {
    if (!runId) {
      console.log('[SENTIMENT-HOOK] getSentimentForRunId: No runId provided, returning null');
      return null;
    }
    
    const sentiment = sentiments[runId] ?? null;
    console.log('[SENTIMENT-HOOK] getSentimentForRunId (DATABASE ONLY):', { runId, sentiment, hasRunId: !!runId });
    
    // Only return sentiment if run_id exists and sentiment is from database
    return sentiment;
  }, [sentiments]);

  return {
    sentiments,
    updateSentiment,
    loadSentiments,
    getSentimentForRunId
  };
} 