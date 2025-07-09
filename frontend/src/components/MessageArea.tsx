import React, { useEffect, useState } from 'react';
import { useSession, getSession } from "next-auth/react";
import Modal from './Modal';
import Link from 'next/link';
import { ChatMessage, ChatThreadMeta, AnalyzeResponse, ChatThreadResponse } from '@/types';
import { API_CONFIG, authApiFetch } from '@/lib/api';
import { useSentiment } from '@/lib/useSentiment';
import { useChatCache } from '@/contexts/ChatCacheContext';

const PROGRESS_DURATION = 480000; // 8 minutes - matches backend analysis timeout

interface SimpleProgressBarProps {
    messageId: number;
    startedAt: number;
}

const SimpleProgressBar = ({ messageId, startedAt }: SimpleProgressBarProps) => {
    const [progress, setProgress] = React.useState(() => {
        const elapsed = Date.now() - startedAt;
        return Math.min(95, (elapsed / PROGRESS_DURATION) * 100); // Cap at 95% until completion
    });
    const [isCompleted, setIsCompleted] = React.useState(false);
    const intervalRef = React.useRef<NodeJS.Timeout | null>(null);

    React.useEffect(() => {
        const update = () => {
            const elapsed = Date.now() - startedAt;
            const percent = Math.min(95, (elapsed / PROGRESS_DURATION) * 100); // Cap at 95% until actual completion
            setProgress(percent);
            
            // Don't auto-complete the progress bar - let the actual response completion do that
            if (elapsed >= PROGRESS_DURATION && intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
        update();
        intervalRef.current = setInterval(update, 1000); // Update every second instead of every 100ms
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [messageId, startedAt]);

    // Complete progress bar when message is done loading
    React.useEffect(() => {
        // Note: This will be triggered when the parent component re-renders with isLoading=false
        // We need a way to detect completion from the parent
        return () => {
            // Cleanup when component unmounts (message stops loading)
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    // Calculate estimated time remaining
    const elapsed = Date.now() - startedAt;
    const remainingMs = Math.max(0, PROGRESS_DURATION - elapsed);
    const remainingMinutes = Math.ceil(remainingMs / 60000);
    const remainingSeconds = Math.ceil((remainingMs % 60000) / 1000);

    return (
        <div className="w-full mt-3">
            <div className="flex justify-between items-center mb-1">
                <span className="text-xs text-gray-500">Processing...</span>
                <span className="text-xs text-gray-500">
                    {remainingMs > 0 ? (
                        remainingMinutes > 0 ? 
                            `~${remainingMinutes}m ${remainingSeconds}s remaining` : 
                            `~${remainingSeconds}s remaining`
                    ) : (
                        'Completing...'
                    )}
                </span>
            </div>
            <div className="h-[3px] w-full bg-gray-200 rounded-full overflow-hidden">
                <div
                    className="h-[3px] bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-1000"
                    style={{ width: `${progress}%` }}
                />
            </div>
        </div>
    );
};

interface FeedbackComponentProps {
    messageId: string;
    runId?: string;
    threadId: string;
    onFeedbackSubmit: (runId: string, feedback: number, comment?: string) => void;
    onCommentSubmit: (runId: string, comment: string) => void;
    feedbackState: { [key: string]: { feedback: number | null; comment?: string } };
    currentSentiment?: boolean | null;
    onSentimentUpdate: (runId: string, sentiment: boolean | null) => void;
}

const FeedbackComponent = ({ messageId, runId, threadId, onFeedbackSubmit, onCommentSubmit, feedbackState, currentSentiment, onSentimentUpdate }: FeedbackComponentProps) => {
    const [showCommentBox, setShowCommentBox] = React.useState(false);
    const [comment, setComment] = React.useState('');
    const [hasProvidedComment, setHasProvidedComment] = React.useState(false);
    const commentButtonRef = React.useRef<HTMLButtonElement>(null);
    const commentBoxRef = React.useRef<HTMLDivElement>(null);
    const messageFeedback = feedbackState[runId || messageId] || { feedback: null, comment: undefined };
    // Local state for persistent feedback
    const [persistentFeedback, setPersistentFeedback] = React.useState<number | null>(null);

    // Load persisted feedback for this specific message on component mount
    React.useEffect(() => {
        const fetchPersistedFeedback = () => {
            try {
                const storageKey = 'czsu-persistent-feedback';
                const savedFeedback = localStorage.getItem(storageKey);
                
                if (savedFeedback) {
                    const feedbackData = JSON.parse(savedFeedback);
                    // Check if we have feedback for this message
                    if (feedbackData[messageId]) {
                        const storedFeedback = feedbackData[messageId].feedbackValue;
                        console.log('[FEEDBACK-STORAGE] Found persisted feedback for message:', 
                            { messageId, feedback: storedFeedback });
                        setPersistentFeedback(storedFeedback);
                    }
                }
            } catch (err) {
                console.error('[FEEDBACK-STORAGE] Error loading persisted feedback for message:', err);
            }
        };
        
        fetchPersistedFeedback();
    }, [messageId]);

    // Click outside to close comment box
    React.useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (showCommentBox && 
                commentBoxRef.current && 
                commentButtonRef.current &&
                !commentBoxRef.current.contains(event.target as Node) &&
                !commentButtonRef.current.contains(event.target as Node)) {
                setShowCommentBox(false);
            }
        };

        if (showCommentBox) {
            document.addEventListener('mousedown', handleClickOutside);
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [showCommentBox]);

    // Save feedback to separate localStorage for persistence
    const saveFeedbackToLocalStorage = (id: string, feedbackValue: number) => {
        try {
            // Use a separate localStorage key that won't be affected by cache invalidations
            const storageKey = 'czsu-persistent-feedback';
            
            // Get existing feedback data or initialize empty object
            const existingData = localStorage.getItem(storageKey);
            const feedbackData = existingData ? JSON.parse(existingData) : {};
            
            // Store feedback data with message ID and run ID (if available)
            feedbackData[messageId] = {
                feedbackValue, 
                timestamp: Date.now(),
                threadId,
                runId: runId || null
            };
            
            // Save back to localStorage
            localStorage.setItem(storageKey, JSON.stringify(feedbackData));
            
            // Update local state
            setPersistentFeedback(feedbackValue);
            
            console.log('[FEEDBACK-STORAGE] Saved feedback to persistent localStorage:', 
                { messageId, runId: runId || null, feedbackValue });
        } catch (err) {
            console.error('[FEEDBACK-STORAGE] Error saving feedback to localStorage:', err);
        }
    };

    const handleFeedback = (feedback: number) => {
        // Use runId if available, otherwise fallback to messageId
        console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleFeedback - using runId:', runId, 'messageId:', messageId);
        
        // Save to separate localStorage for persistence
        saveFeedbackToLocalStorage(runId || messageId, feedback);
        
        // Update sentiment if we have a runId (new sentiment system)
        if (runId) {
            const sentiment = feedback === 1 ? true : false;
            onSentimentUpdate(runId, sentiment);
        }
        
        // Call the original onFeedbackSubmit function (existing LangSmith feedback)
        onFeedbackSubmit(runId || messageId, feedback);
        setShowCommentBox(false);
        setComment('');
    };

    const handleCommentSubmit = () => {
        // Use runId if available, otherwise fallback to messageId
        console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleCommentSubmit - using runId:', runId, 'messageId:', messageId);
        const feedbackValue = messageFeedback.feedback !== null ? messageFeedback.feedback : 1;
        
        // Save to localStorage along with comment
        saveFeedbackToLocalStorage(runId || messageId, feedbackValue);
        
        // Call the comment submit function from MessageArea
        if (runId && comment.trim()) {
            onCommentSubmit(runId, comment.trim());
        }
        
        setShowCommentBox(false);
        setHasProvidedComment(true); // Mark that a comment was provided
        setComment('');
    };

    // Comment icon with checkmark overlay when comment provided
    const CommentIcon = () => {
        if (hasProvidedComment) {
            return (
                <div className="relative">
                    <span>💬</span>
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full flex items-center justify-center">
                        <span className="text-white text-xs leading-none">✓</span>
                    </div>
                </div>
            );
        }
        return <span>💬</span>;
    };

    // Use either API feedback state or persistent localStorage feedback
    const effectiveFeedbackValue = messageFeedback.feedback !== null ? messageFeedback.feedback : persistentFeedback;

    return (
        <div className="flex items-center space-x-2 relative">
            {/* Show "selected" message if sentiment is already chosen */}
            {currentSentiment !== null ? (
                <div className="flex items-center space-x-1 px-2 py-1 rounded bg-blue-50 text-blue-700 text-sm font-medium">
                    <span>selected:</span>
                    <span>{currentSentiment === true ? '👍' : '👎'}</span>
                </div>
            ) : (
                // Show clickable thumbs if no sentiment is selected
                <>
                    {/* Thumbs up */}
                    <button
                        onClick={() => handleFeedback(1)}
                        className="p-1 rounded transition-colors text-gray-400 hover:text-blue-600 hover:bg-blue-50"
                        title="Good response"
                    >
                        👍
                    </button>
                    
                    {/* Thumbs down */}
                    <button
                        onClick={() => handleFeedback(0)}
                        className="p-1 rounded transition-colors text-gray-400 hover:text-blue-600 hover:bg-blue-50"
                        title="Poor response"
                    >
                        👎
                    </button>
                </>
            )}
            
            {/* Comment button with fixed positioning context */}
            <div className="relative">
                <button
                    ref={commentButtonRef}
                    onClick={() => setShowCommentBox(!showCommentBox)}
                    className={`p-1 rounded transition-colors ${
                        showCommentBox 
                            ? 'text-blue-600 bg-blue-50' 
                            : hasProvidedComment
                            ? 'text-green-600 hover:text-green-700 hover:bg-green-50'
                            : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                    }`}
                    title={hasProvidedComment ? "Comment provided - click to edit" : "Add comment"}
                >
                    <CommentIcon />
                </button>
                
                {/* Comment box - positioned relative to comment button wrapper */}
                {showCommentBox && (
                    <div 
                        ref={commentBoxRef}
                        className="absolute bottom-full right-0 mb-2 p-3 bg-white border border-gray-200 rounded-lg shadow-lg min-w-[300px] z-20"
                    >
                        <textarea
                            value={comment}
                            onChange={(e) => setComment(e.target.value)}
                            placeholder="Share your feedback..."
                            className="w-full p-2 border border-gray-300 rounded text-sm resize-none"
                            rows={3}
                            autoFocus
                        />
                        <div className="flex justify-end space-x-2 mt-2">
                            <button
                                onClick={() => setShowCommentBox(false)}
                                className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleCommentSubmit}
                                className="px-4 py-2 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                disabled={!comment.trim()}
                            >
                                Submit
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

interface MessageAreaProps {
    messages: any[];
    threadId: string | null;
    onSQLClick: (msgId: string) => void;
    openSQLModalForMsgId: string | null;
    onCloseSQLModal: () => void;
    onPDFClick: (msgId: string) => void;
    openPDFModalForMsgId: string | null;
    onClosePDFModal: () => void;
    onNewChat: () => void;
    isLoading: boolean;
    isAnyLoading?: boolean;
    threads?: any[];
    activeThreadId?: string | null;
}

// Add a global utility function to get all persisted feedback
const getPersistedFeedbackData = (): { [key: string]: any } => {
    try {
        const storageKey = 'czsu-persistent-feedback';
        const savedFeedback = localStorage.getItem(storageKey);
        return savedFeedback ? JSON.parse(savedFeedback) : {};
    } catch (err) {
        console.error('[FEEDBACK-STORAGE] Error getting persisted feedback data:', err);
        return {};
    }
};

const MessageArea = ({ messages, threadId, onSQLClick, openSQLModalForMsgId, onCloseSQLModal, onPDFClick, openPDFModalForMsgId, onClosePDFModal, onNewChat, isLoading, isAnyLoading, threads, activeThreadId }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    const containerRef = React.useRef<HTMLDivElement>(null);
    
    // State for feedback functionality
    const [feedbackState, setFeedbackState] = React.useState<{ [runId: string]: { feedback: number | null; comment?: string } }>({});
    const [messageRunIds, setMessageRunIds] = React.useState<{[messageId: string]: string}>({});
    
    // NEW: Use ChatCacheContext for cached data access
    const { getRunIdsForThread, getSentimentsForThread, updateCachedSentiment } = useChatCache();
    
    // Use sentiments from cache instead of API calls
    const { sentiments, updateSentiment, loadSentiments, getSentimentForRunId } = useSentiment();
    
    // Track which run_ids have already sent feedback to LangSmith to prevent duplicates
    const [langsmithFeedbackSent, setLangsmithFeedbackSent] = React.useState<Set<string>>(new Set());
    
    // Auto-scroll to bottom when messages change or thread changes
    React.useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, threadId]);

    // Reset LangSmith feedback tracking when thread changes
    React.useEffect(() => {
        setLangsmithFeedbackSent(new Set());
    }, [threadId]);

    // Load run-ids from cache instead of making API calls
    React.useEffect(() => {
        const loadRunIdsFromCache = () => {
            if (!threadId) return;
            
            console.log('[FEEDBACK-DEBUG] Loading run_ids from cache for thread:', threadId);
            
            // Get cached run-ids for this thread
            const cachedRunIds = getRunIdsForThread(threadId);
            console.log('[FEEDBACK-DEBUG] Found cached run_ids:', cachedRunIds.length);
            
            if (cachedRunIds && cachedRunIds.length > 0) {
                const newMessageRunIds: {[messageId: string]: string} = {};
                
                // Get all non-user messages (AI responses) in order
                const aiMessages = messages.filter(message => !message.isUser);
                
                // For each AI message, try to find a matching run_id
                aiMessages.forEach((message, index) => {
                    // Skip if message already has run_id in meta
                    if (message.meta?.runId) {
                        console.log('[FEEDBACK-DEBUG] Message already has run_id in meta:', message.meta.runId);
                        newMessageRunIds[message.id] = message.meta.runId;
                        return;
                    }
                    
                    // Match by index (first AI message gets first run_id, etc.)
                    if (index < cachedRunIds.length) {
                        const runIdEntry = cachedRunIds[index];
                        newMessageRunIds[message.id] = runIdEntry.run_id;
                        console.log('[FEEDBACK-DEBUG] Assigned cached run_id to message by index:', 
                            {messageId: message.id, runId: runIdEntry.run_id, index});
                    } else {
                        console.log('[FEEDBACK-DEBUG] No cached run_id available for message at index:', index);
                    }
                });
                
                setMessageRunIds(newMessageRunIds);
            } else {
                console.log('[FEEDBACK-DEBUG] No cached run_ids found for thread:', threadId);
            }
        };
        
        loadRunIdsFromCache();
    }, [threadId, messages, getRunIdsForThread]);

    // OPTIMIZED: Sentiments are loaded via bulk loading - no individual loading needed
    React.useEffect(() => {
        if (!threadId) return;
        
        console.log('[SENTIMENT-DEBUG] Thread changed to:', threadId);
        console.log('[SENTIMENT-DEBUG] Sentiments are automatically loaded via bulk loading - no individual API calls needed');
        
        // Get cached sentiments for this thread to verify they're available
        const cachedSentiments = getSentimentsForThread(threadId);
        console.log('[SENTIMENT-DEBUG] Found cached sentiments for thread:', Object.keys(cachedSentiments).length);
        
        // No need to call loadSentiments() - bulk loading handles everything
    }, [threadId, getSentimentsForThread]);

    // Session and authentication
    const { data: session } = useSession();
    const userEmail = session?.user?.email || null;

    const handleFeedbackSubmit = async (runId: string, feedback: number, comment?: string) => {
        console.log('[FEEDBACK-DEBUG] handleFeedbackSubmit called:', JSON.stringify({ runId, feedback, comment }));
        console.log('[FEEDBACK-DEBUG] runId type:', typeof runId, 'runId value:', `"${runId}"`, 'runId length:', runId ? runId.length : 0);
        
        if (!runId) {
            console.log('[FEEDBACK-DEBUG] Skipping feedback submit: missing runId');
            return;
        }
        
        // Check if we've already sent feedback to LangSmith for this run_id
        if (langsmithFeedbackSent.has(runId) && !comment) {
            console.log('[FEEDBACK-DEBUG] Skipping LangSmith feedback submit: already sent for runId:', runId);
            return;
        }
        
        try {
            const freshSession = await getSession();
            if (!freshSession?.id_token) {
                console.log('[FEEDBACK-DEBUG] No id_token available for feedback submit');
                return;
            }
            
            // Check if this looks like a UUID before sending
            const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
            if (!uuidPattern.test(runId)) {
                console.log('[FEEDBACK-DEBUG] ⚠️ Warning: runId does not appear to be a valid UUID format:', runId);
            }
            
            console.log('[FEEDBACK-DEBUG] Sending feedback to LangSmith:', JSON.stringify({ run_id: runId, feedback, comment }));
            
            try {
                const body: any = { run_id: runId, feedback };
                if (comment) {
                    body.comment = comment;
                }
                
                await authApiFetch('/feedback', freshSession.id_token, {
                    method: 'POST',
                    body: JSON.stringify(body)
                });
                console.log('[FEEDBACK-DEBUG] Feedback submitted successfully to LangSmith for runId:', runId);
                
                // Mark this run_id as having sent feedback to LangSmith
                setLangsmithFeedbackSent(prev => new Set([...prev, runId]));
                
                // Update the feedback state 
                setFeedbackState(prev => ({ 
                    ...prev, 
                    [runId]: { 
                        ...prev[runId], 
                        feedback,
                        ...(comment && { comment })
                    } 
                }));
            } catch (fetchError) {
                console.error('[FEEDBACK-DEBUG] Fetch error details:', fetchError);
                if (fetchError instanceof Response) {
                    const errorText = await fetchError.text();
                    console.error('[FEEDBACK-DEBUG] Error response body:', errorText);
                }
                throw fetchError; // Re-throw to be caught by outer catch
            }
        } catch (error) {
            console.error('[FEEDBACK-DEBUG] Error submitting feedback:', error);
            console.error('[FEEDBACK-DEBUG] For runId:', runId, 'feedback:', feedback, 'comment:', comment);
            if (error instanceof Error) {
                console.error('[FEEDBACK-DEBUG] Error name:', error.name, 'message:', error.message);
            }
        }
    };

    const handleCommentSubmit = async (runId: string, comment: string) => {
        console.log('[FEEDBACK-DEBUG] handleCommentSubmit called:', JSON.stringify({ 
            runId, 
            comment, 
            feedback: feedbackState[runId]?.feedback 
        }));
        console.log('[FEEDBACK-DEBUG] Comment runId type:', typeof runId, 'value:', `"${runId}"`);
        
        if (!runId) {
            console.log('[FEEDBACK-DEBUG] Skipping comment submit: missing runId');
            return;
        }
        
        try {
            const freshSession = await getSession();
            if (!freshSession?.id_token) {
                console.log('[FEEDBACK-DEBUG] No id_token available for comment submit');
                return;
            }
            
            // Check if this looks like a UUID before sending
            const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
            if (!uuidPattern.test(runId)) {
                console.log('[FEEDBACK-DEBUG] ⚠️ Warning: comment runId does not appear to be a valid UUID format:', runId);
            }
            
            // Only send score if user actually provided one (clicked thumbs up/down)
            const existingFeedback = feedbackState[runId]?.feedback;
            const hasScore = existingFeedback !== null && existingFeedback !== undefined;
            
            console.log('[FEEDBACK-DEBUG] Sending comment to LangSmith:', JSON.stringify({ 
                run_id: runId, 
                feedback: hasScore ? existingFeedback : 'no score provided', 
                comment,
                hasScore
            }));
            
            try {
                const body: any = { run_id: runId, comment };
                
                // Only include feedback score if user actually provided one
                if (hasScore) {
                    body.feedback = existingFeedback;
                }
                
                await authApiFetch('/feedback', freshSession.id_token, {
                    method: 'POST',
                    body: JSON.stringify(body)
                });
                console.log('[FEEDBACK-DEBUG] Comment submitted successfully to LangSmith for runId:', runId);
                
                // Mark this run_id as having sent feedback to LangSmith
                setLangsmithFeedbackSent(prev => new Set([...prev, runId]));
                
                setFeedbackState(prev => ({ 
                    ...prev, 
                    [runId]: { 
                        ...prev[runId], 
                        ...(hasScore && { feedback: existingFeedback }),
                        comment 
                    } 
                }));
            } catch (fetchError) {
                console.error('[FEEDBACK-DEBUG] Comment fetch error details:', fetchError);
                if (fetchError instanceof Response) {
                    const errorText = await fetchError.text();
                    console.error('[FEEDBACK-DEBUG] Comment error response body:', errorText);
                }
                throw fetchError;
            }
        } catch (error) {
            console.error('[FEEDBACK-DEBUG] Error submitting comment:', error);
            console.error('[FEEDBACK-DEBUG] For runId:', runId, 'comment length:', comment?.length);
            if (error instanceof Error) {
                console.error('[FEEDBACK-DEBUG] Comment error name:', error.name, 'message:', error.message);
            }
        }
    };

    return (
        <div 
            ref={containerRef} 
            className="h-full overflow-y-auto bg-gradient-to-br from-white to-blue-50/20 chat-scrollbar" 
            style={{ minHeight: 0 }} 
            aria-live="polite" 
            role="log"
        >
            <div className="max-w-4xl mx-auto p-6">
                {messages.length === 0 ? (
                    <div className="flex items-center justify-center h-full min-h-[400px]">
                        <div className="text-center">
                            <div className="text-6xl mb-4">💬</div>
                            <h3 className="text-xl font-semibold text-gray-700 mb-2">Start a conversation</h3>
                            <p className="text-gray-500">Ask me about your data and I'll help you analyze it!</p>
                        </div>
                    </div>
                ) : (
                    messages.map((message) => {
                        // Determine if message is from user based on presence of prompt
                        const isUser = !!message.prompt;
                        const content = isUser ? message.prompt : message.final_answer;
                        
                        return (
                            <div key={message.id} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
                                <div className="flex flex-col max-w-2xl w-full">
                                    {/* Message Content */}
                                    <div
                                        className={`transition-all duration-200 rounded-2xl px-6 py-4 w-full select-text shadow-lg group
                                            ${isUser
                                                ? 'light-blue-theme font-semibold hover:shadow-xl'
                                                : message.isError
                                                    ? 'bg-red-50 border border-red-200 text-red-800 hover:shadow-xl hover:border-red-300'
                                                    : 'bg-white border border-blue-100 text-gray-800 hover:shadow-xl hover:border-blue-200'}
                                        `}
                                        style={{ 
                                            fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)', 
                                            fontSize: '0.97rem', 
                                            lineHeight: 1.6, 
                                            wordBreak: 'break-word', 
                                            whiteSpace: 'pre-line' 
                                        }}
                                    >
                                        {message.isLoading && !content ? (
                                            <div className="flex items-center space-x-3">
                                                <div className="w-5 h-5 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin"></div>
                                                <span className="text-gray-600">Analyzing your request...</span>
                                            </div>
                                        ) : (
                                            content || (
                                                <span className="text-gray-400 text-xs italic">Waiting for response...</span>
                                            )
                                        )}
                                    </div>

                                    {/* Dataset used and SQL button for AI answers */}
                                    {!isUser && !message.isLoading && (message.datasets_used?.length || message.sql_query || message.top_chunks?.length) && (
                                        <div className="mt-3 flex items-center justify-between flex-wrap" style={{ fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)' }}>
                                            <div className="flex items-center space-x-3 flex-wrap">
                                                {/* Show datasets */}
                                                {message.datasets_used && message.datasets_used.length > 0 && (
                                                    <div className="flex items-center space-x-2 flex-wrap">
                                                        <span className="text-xs text-gray-500 mr-1" style={{ marginLeft: '1rem' }}>Dataset{message.datasets_used.length > 1 ? 's' : ''} used:</span>
                                                        {message.datasets_used.map((code: string, index: number) => (
                                                            <Link
                                                                key={index}
                                                                href={`/data?table=${encodeURIComponent(code)}`}
                                                                className="inline-block px-3 py-1 rounded-full bg-blue-50 text-blue-700 font-mono text-xs font-semibold hover:bg-blue-100 transition-all duration-150 shadow-sm border border-blue-100"
                                                                style={{ textDecoration: 'none' }}
                                                            >
                                                                {code}
                                                            </Link>
                                                        ))}
                                                    </div>
                                                )}

                                                {/* Action Buttons */}
                                                {message.sql_query && (
                                                    <button
                                                        className="px-4 py-1 rounded-full light-blue-theme text-xs font-bold transition-all duration-150"
                                                        onClick={() => onSQLClick(message.id)}
                                                    >
                                                        SQL
                                                    </button>
                                                )}
                                                {message.top_chunks && message.top_chunks.length > 0 && (
                                                    <button
                                                        className="px-4 py-1 rounded-full light-blue-theme text-xs font-bold transition-all duration-150"
                                                        onClick={() => onPDFClick(message.id)}
                                                    >
                                                        PDF
                                                    </button>
                                                )}
                                            </div>

                                            {/* Feedback component */}
                                            {threadId && (
                                                <FeedbackComponent
                                                    messageId={message.id}
                                                    runId={messageRunIds[message.id]}
                                                    threadId={threadId}
                                                    onFeedbackSubmit={handleFeedbackSubmit}
                                                    onCommentSubmit={handleCommentSubmit}
                                                    feedbackState={feedbackState}
                                                    currentSentiment={getSentimentForRunId(messageRunIds[message.id] || '')}
                                                    onSentimentUpdate={updateSentiment}
                                                />
                                            )}
                                        </div>
                                    )}

                                    {/* Show feedback even when no datasets/SQL */}
                                    {!isUser && !message.isLoading && threadId && !(message.datasets_used?.length || message.sql_query || message.top_chunks?.length) && (
                                        <div className="mt-3 flex justify-end">
                                            <FeedbackComponent
                                                messageId={message.id}
                                                runId={messageRunIds[message.id]}
                                                threadId={threadId}
                                                onFeedbackSubmit={handleFeedbackSubmit}
                                                onCommentSubmit={handleCommentSubmit}
                                                feedbackState={feedbackState}
                                                currentSentiment={getSentimentForRunId(messageRunIds[message.id] || '')}
                                                onSentimentUpdate={updateSentiment}
                                            />
                                        </div>
                                    )}

                                    {/* SQL Modal */}
                                    {openSQLModalForMsgId === message.id && (
                                        <Modal open={true} onClose={onCloseSQLModal}>
                                            <h2 className="text-lg font-bold mb-4">SQL Commands & Results</h2>
                                            <div className="max-h-[60vh] overflow-y-auto pr-2 chat-scrollbar">
                                                {(() => {
                                                    const uniqueQueriesAndResults = Array.from(
                                                        new Map((message.queries_and_results || []).map(([q, r]: [string, string]) => [q, [q, r]])).values()
                                                    ) as [string, string][];
                                                    if (uniqueQueriesAndResults.length === 0) {
                                                        return <div className="text-gray-500">No SQL commands available.</div>;
                                                    }
                                                    return (
                                                        <div className="space-y-6">
                                                            {uniqueQueriesAndResults.map(([sql, result]: [string, string], idx: number) => (
                                                                <div key={idx} className="bg-gray-50 rounded border border-gray-200 p-0">
                                                                    <div className="bg-gray-100 px-4 py-2 rounded-t text-xs font-semibold text-gray-700 border-b border-gray-200">SQL Command {idx + 1}</div>
                                                                    <div className="p-3 font-mono text-xs whitespace-pre-line text-gray-900">
                                                                        {sql.split('\n').map((line: string, i: number) => (
                                                                            <React.Fragment key={i}>
                                                                                {line}
                                                                                {i !== sql.split('\n').length - 1 && <br />}
                                                                            </React.Fragment>
                                                                        ))}
                                                                    </div>
                                                                    <div className="bg-gray-100 px-4 py-2 text-xs font-semibold text-gray-700 border-t border-gray-200">Result</div>
                                                                    <div className="p-3 font-mono text-xs whitespace-pre-line text-gray-800">
                                                                        {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    );
                                                })()}
                                            </div>
                                        </Modal>
                                    )}

                                    {/* PDF Modal */}
                                    {openPDFModalForMsgId === message.id && (
                                        <Modal open={true} onClose={onClosePDFModal}>
                                            <h2 className="text-lg font-bold mb-4">PDF Document Chunks</h2>
                                            <div className="max-h-[60vh] overflow-y-auto pr-2 chat-scrollbar">
                                                {(() => {
                                                    const chunks = message.top_chunks || [];
                                                    if (chunks.length === 0) {
                                                        return <div className="text-gray-500">No PDF chunks available.</div>;
                                                    }
                                                    return (
                                                        <div className="space-y-6">
                                                            {chunks.map((chunk: any, idx: number) => (
                                                                <div key={idx} className="bg-gray-50 rounded border border-gray-200 p-0">
                                                                    <div className="bg-gray-100 px-4 py-2 rounded-t text-xs font-semibold text-gray-700 border-b border-gray-200">
                                                                        PDF Chunk {idx + 1}
                                                                        {chunk.source_file && (
                                                                            <span className="ml-2 font-normal text-gray-600">
                                                                                , {chunk.source_file}
                                                                            </span>
                                                                        )}
                                                                        {chunk.page_number && (
                                                                            <span className="ml-1 font-normal text-gray-600">
                                                                                , page {chunk.page_number}
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                    <div className="p-3 text-xs whitespace-pre-line text-gray-800 leading-relaxed">
                                                                        {chunk.page_content}
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    );
                                                })()}
                                            </div>
                                        </Modal>
                                    )}
                                </div>
                            </div>
                        );
                    })
                )}
                
                {/* New Chat Button at bottom of scrollable content */}
                <div className="flex justify-center py-6">
                    <button
                        className="px-4 py-2 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={onNewChat}
                        title="Start a new chat"
                        disabled={isAnyLoading}
                    >
                        + New Chat
                    </button>
                </div>
                
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default MessageArea;