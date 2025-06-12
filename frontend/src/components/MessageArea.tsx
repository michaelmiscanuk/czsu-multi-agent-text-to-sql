import React, { useEffect, useState } from 'react';
import { useSession, getSession } from "next-auth/react";
import Modal from './Modal';
import Link from 'next/link';
import { ChatMessage } from '@/types';
import { API_CONFIG, authApiFetch } from '@/lib/api';

const PROGRESS_DURATION = 20000; // 20 seconds

interface SimpleProgressBarProps {
    messageId: number;
    startedAt: number;
}

const SimpleProgressBar = ({ messageId, startedAt }: SimpleProgressBarProps) => {
    const [progress, setProgress] = React.useState(() => {
        const elapsed = Date.now() - startedAt;
        return Math.min(100, (elapsed / PROGRESS_DURATION) * 100);
    });
    const intervalRef = React.useRef<NodeJS.Timeout | null>(null);

    React.useEffect(() => {
        const update = () => {
            const elapsed = Date.now() - startedAt;
            const percent = Math.min(100, (elapsed / PROGRESS_DURATION) * 100);
            setProgress(percent);
            if (percent >= 100 && intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
        update();
        intervalRef.current = setInterval(update, 100);
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [messageId, startedAt]);

    return (
        <div className="w-full mt-2">
            <div className="h-[3px] w-full bg-gray-200 rounded-full overflow-hidden">
                <div
                    className="h-[3px] bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-200"
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
    feedbackState: { [key: string]: { feedback: number | null; comment?: string } };
}

const FeedbackComponent = ({ messageId, runId, threadId, onFeedbackSubmit, feedbackState }: FeedbackComponentProps) => {
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
        
        // Call the original onFeedbackSubmit function
        onFeedbackSubmit(runId || messageId, feedback, comment || undefined);
        setShowCommentBox(false);
        setComment('');
    };

    const handleCommentSubmit = () => {
        // Use runId if available, otherwise fallback to messageId
        console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleCommentSubmit - using runId:', runId, 'messageId:', messageId);
        const feedbackValue = messageFeedback.feedback !== null ? messageFeedback.feedback : 1;
        
        // Save to localStorage along with comment
        saveFeedbackToLocalStorage(runId || messageId, feedbackValue);
        
        // Call the original onFeedbackSubmit function
        onFeedbackSubmit(runId || messageId, feedbackValue, comment || undefined);
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
            {/* Thumbs up */}
            <button
                onClick={() => handleFeedback(1)}
                className={`p-1 rounded transition-colors ${
                    effectiveFeedbackValue === 1 
                        ? 'text-white bg-blue-500 hover:bg-blue-600 shadow-md' 
                        : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                }`}
                title="Good response"
            >
                👍
            </button>
            
            {/* Thumbs down */}
            <button
                onClick={() => handleFeedback(0)}
                className={`p-1 rounded transition-colors ${
                    effectiveFeedbackValue === 0 
                        ? 'text-white bg-blue-500 hover:bg-blue-600 shadow-md' 
                        : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                }`}
                title="Poor response"
            >
                👎
            </button>
            
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

const MessageArea = ({ messages, threadId, onSQLClick, openSQLModalForMsgId, onCloseSQLModal, onNewChat, isLoading, isAnyLoading, threads, activeThreadId }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    const containerRef = React.useRef<HTMLDivElement>(null);
    
    // State for feedback functionality
    const [feedbackState, setFeedbackState] = React.useState<{ [runId: string]: { feedback: number | null; comment?: string } }>({});
    const [messageRunIds, setMessageRunIds] = React.useState<{[messageId: string]: string}>({});
    
    // Auto-scroll to bottom when messages change or thread changes
    React.useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, threadId]);

    // Fetch run_ids for the current thread when it changes
    React.useEffect(() => {
        const fetchRunIds = async () => {
            if (!threadId) return;
            
            try {
                console.log('[FEEDBACK-DEBUG] Fetching run_ids for thread:', threadId);
                const freshSession = await getSession();
                if (!freshSession?.id_token) {
                    console.log('[FEEDBACK-DEBUG] No id_token available for run_id fetch');
                    return;
                }
                
                const data = await authApiFetch<{run_ids: Array<{run_id: string, prompt: string, timestamp: string}>}>(`/chat/${threadId}/run-ids`, freshSession.id_token);
                console.log('[FEEDBACK-DEBUG] Received run_ids:', JSON.stringify(data.run_ids));
                
                // Match run_ids with messages based on content/prompt similarity
                if (data.run_ids && data.run_ids.length > 0) {
                    const newMessageRunIds: {[messageId: string]: string} = {};
                    
                    // For each non-user message, try to find a matching run_id
                    messages.forEach(message => {
                        if (!message.isUser) {
                            // Skip if message already has run_id in meta
                            if (message.meta?.run_id) {
                                console.log('[FEEDBACK-DEBUG] Message already has run_id in meta:', message.meta.run_id);
                                newMessageRunIds[message.id] = message.meta.run_id;
                                return;
                            }
                            
                            // Otherwise try to match by finding the closest run_id by timestamp
                            // This is a simplistic approach - in a real system we would have better matching
                            if (data.run_ids.length > 0) {
                                const runIdEntry = data.run_ids[data.run_ids.length - 1]; // Use the last run_id
                                newMessageRunIds[message.id] = runIdEntry.run_id;
                                console.log('[FEEDBACK-DEBUG] Assigned run_id to message:', 
                                    {messageId: message.id, runId: runIdEntry.run_id});
                            }
                        }
                    });
                    
                    setMessageRunIds(newMessageRunIds);
                }
            } catch (error) {
                console.error('[FEEDBACK-DEBUG] Error fetching run_ids:', error);
            }
        };
        
        fetchRunIds();
    }, [threadId, messages]);

    // Session and authentication
    const { data: session } = useSession();
    const userEmail = session?.user?.email || null;

    const handleFeedbackSubmit = async (runId: string, feedback: number) => {
        console.log('[FEEDBACK-DEBUG] handleFeedbackSubmit called:', JSON.stringify({ runId, feedback }));
        console.log('[FEEDBACK-DEBUG] runId type:', typeof runId, 'runId value:', `"${runId}"`, 'runId length:', runId ? runId.length : 0);
        
        if (!runId) {
            console.log('[FEEDBACK-DEBUG] Skipping feedback submit: missing runId');
            return;
        }
        
        if (feedbackState[runId]?.feedback !== undefined) {
            console.log('[FEEDBACK-DEBUG] Skipping feedback submit: already submitted for runId:', runId);
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
            
            console.log('[FEEDBACK-DEBUG] Sending feedback to backend:', JSON.stringify({ run_id: runId, feedback }));
            
            try {
                await authApiFetch('/feedback', freshSession.id_token, {
                    method: 'POST',
                    body: JSON.stringify({ run_id: runId, feedback })
                });
                console.log('[FEEDBACK-DEBUG] Feedback submitted successfully for runId:', runId);
                
                // Update the feedback state 
                setFeedbackState(prev => ({ ...prev, [runId]: { ...prev[runId], feedback } }));
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
            console.error('[FEEDBACK-DEBUG] For runId:', runId, 'feedback:', feedback);
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
        
        if (feedbackState[runId]?.comment) {
            console.log('[FEEDBACK-DEBUG] Skipping comment submit: already commented for runId:', runId);
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
            
            console.log('[FEEDBACK-DEBUG] Sending comment to backend:', JSON.stringify({ 
                run_id: runId, 
                feedback: feedbackState[runId]?.feedback ?? null, 
                comment 
            }));
            
            try {
                await authApiFetch('/feedback', freshSession.id_token, {
                    method: 'POST',
                    body: JSON.stringify({ 
                        run_id: runId, 
                        feedback: feedbackState[runId]?.feedback ?? null, 
                        comment 
                    })
                });
                console.log('[FEEDBACK-DEBUG] Comment submitted successfully for runId:', runId);
                setFeedbackState(prev => ({ ...prev, [runId]: { ...prev[runId], comment } }));
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
                    messages.map((message) => (
                        <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-6`}>
                            <div className="flex flex-col max-w-2xl w-full">
                                {/* Message Content */}
                                <div
                                    className={`transition-all duration-200 rounded-2xl px-6 py-4 w-full select-text shadow-lg group
                                        ${message.isUser
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
                                    {message.isLoading && !message.content ? (
                                        <div className="flex items-center space-x-3">
                                            <div className="w-5 h-5 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin"></div>
                                            <span className="text-gray-600">Thinking...</span>
                                        </div>
                                    ) : (
                                        message.content || (
                                            // Fallback if content is empty but not in loading state
                                            <span className="text-gray-400 text-xs italic">Waiting for response...</span>
                                        )
                                    )}
                                    {message.isUser ? null : console.log('[FEEDBACK-DEBUG] Message meta:', JSON.stringify({
                                        id: message.id,
                                        has_meta: !!message.meta,
                                        run_id: message.meta?.run_id || 'none',
                                        meta_keys: message.meta ? Object.keys(message.meta) : []
                                    }))}
                                </div>
                                {/* Dataset used and SQL button for AI answers */}
                                {!message.isUser && !message.isLoading && (message.selectionCode || message.meta?.datasetUrl || message.meta?.datasetsUsed?.length || message.meta?.sqlQuery) && (
                                    <div className="mt-3 flex items-center justify-between flex-wrap" style={{ fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)' }}>
                                        <div className="flex items-center space-x-3 flex-wrap">
                                            {/* Show multiple dataset codes if available */}
                                            {message.meta?.datasetsUsed && message.meta.datasetsUsed.length > 0 ? (
                                                <div className="flex items-center space-x-2 flex-wrap">
                                                    <span className="text-xs text-gray-500 mr-1">Dataset{message.meta.datasetsUsed.length > 1 ? 's' : ''} used:</span>
                                                    {message.meta.datasetsUsed.map((code: string, index: number) => (
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
                                            ) : (
                                                /* Fallback to old single dataset approach for backward compatibility */
                                                (message.selectionCode || message.meta?.datasetUrl) && (
                                                    <div>
                                                        <span className="text-xs text-gray-500 mr-1">Dataset used:</span>
                                                        <Link
                                                            href={`/data?table=${encodeURIComponent(message.selectionCode || message.meta?.datasetUrl.replace('/datasets/', ''))}`}
                                                            className="inline-block px-3 py-1 rounded-full bg-blue-50 text-blue-700 font-mono text-xs font-semibold hover:bg-blue-100 transition-all duration-150 shadow-sm border border-blue-100"
                                                            style={{ textDecoration: 'none' }}
                                                        >
                                                            {message.selectionCode || (message.meta?.datasetUrl ? message.meta.datasetUrl.replace('/datasets/', '') : '')}
                                                        </Link>
                                                    </div>
                                                )
                                            )}
                                            {message.meta?.sqlQuery && (
                                                <button
                                                    className="px-4 py-1 rounded-full light-blue-theme text-xs font-bold transition-all duration-150"
                                                    onClick={() => onSQLClick(message.id)}
                                                >
                                                    SQL
                                                </button>
                                            )}
                                        </div>
                                        
                                        {/* Feedback component aligned to the right */}
                                        {threadId && (
                                            <FeedbackComponent
                                                messageId={message.id}
                                                runId={message.meta?.run_id || messageRunIds[message.id]}
                                                threadId={threadId}
                                                onFeedbackSubmit={handleFeedbackSubmit}
                                                feedbackState={feedbackState}
                                            />
                                        )}
                                        
                                        {/* SQL Modal for this message */}
                                        {openSQLModalForMsgId === message.id && (
                                            <Modal open={true} onClose={onCloseSQLModal}>
                                                <h2 className="text-lg font-bold mb-4">SQL Commands & Results</h2>
                                                <div className="max-h-[60vh] overflow-y-auto pr-2 chat-scrollbar">
                                                    {(() => {
                                                        const uniqueQueriesAndResults = Array.from(
                                                            new Map((message.queriesAndResults || []).map(([q, r]: [string, string]) => [q, [q, r]])).values()
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
                                    </div>
                                )}
                                
                                {/* Show feedback component even when no datasets/SQL - for messages without metadata */}
                                {!message.isUser && !message.isLoading && threadId && !(message.selectionCode || message.meta?.datasetUrl || message.meta?.datasetsUsed?.length || message.meta?.sqlQuery) && (
                                    <div className="mt-3 flex justify-end">
                                        <FeedbackComponent
                                            messageId={message.id}
                                            runId={message.meta?.run_id || messageRunIds[message.id]}
                                            threadId={threadId}
                                            onFeedbackSubmit={handleFeedbackSubmit}
                                            feedbackState={feedbackState}
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
                
                {/* New Chat Button at bottom of scrollable content */}
                <div className="flex justify-center py-6">
                    <button
                        className="px-4 py-2 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={onNewChat}
                        title="Start a new chat"
                        disabled={isAnyLoading || (threads && threads.some(s => !messages.length && s.thread_id === activeThreadId))}
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