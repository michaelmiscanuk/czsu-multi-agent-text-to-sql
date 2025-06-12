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
    threadId: string;
    onFeedbackSubmit: (messageId: string, feedback: number, comment?: string) => void;
    feedbackState: { [key: string]: { feedback: number | null; hasSubmitted: boolean } };
}

const FeedbackComponent = ({ messageId, threadId, onFeedbackSubmit, feedbackState }: FeedbackComponentProps) => {
    const [showCommentBox, setShowCommentBox] = React.useState(false);
    const [comment, setComment] = React.useState('');
    const commentButtonRef = React.useRef<HTMLButtonElement>(null);
    const messageFeedback = feedbackState[messageId] || { feedback: null, hasSubmitted: false };

    const handleFeedback = (feedback: number) => {
        if (messageFeedback.hasSubmitted) return; // Prevent multiple submissions
        onFeedbackSubmit(messageId, feedback, comment || undefined);
        setShowCommentBox(false);
        setComment('');
    };

    const handleCommentSubmit = () => {
        if (messageFeedback.hasSubmitted) return;
        // Submit with last selected feedback or default to neutral
        const feedbackValue = messageFeedback.feedback !== null ? messageFeedback.feedback : 1;
        onFeedbackSubmit(messageId, feedbackValue, comment || undefined);
        setShowCommentBox(false);
        setComment('');
    };

    if (messageFeedback.hasSubmitted) {
        // Show submitted state
        return (
            <div className="flex items-center space-x-2 text-xs text-gray-500">
                <span>
                    {messageFeedback.feedback === 1 ? '👍' : '👎'} Feedback submitted
                </span>
            </div>
        );
    }

    return (
        <div className="flex items-center space-x-2 relative">
            {/* Thumbs up */}
            <button
                onClick={() => handleFeedback(1)}
                className={`p-1 rounded transition-colors ${
                    messageFeedback.feedback === 1 
                        ? 'text-blue-600 bg-blue-50' 
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
                    messageFeedback.feedback === 0 
                        ? 'text-blue-600 bg-blue-50' 
                        : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                }`}
                title="Poor response"
            >
                👎
            </button>
            
            {/* Comment button */}
            <button
                ref={commentButtonRef}
                onClick={() => setShowCommentBox(!showCommentBox)}
                className={`p-1 rounded transition-colors ${
                    showCommentBox 
                        ? 'text-blue-600 bg-blue-50' 
                        : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                }`}
                title="Add comment"
            >
                💬
            </button>
            
            {/* Comment box - positioned above the comment icon */}
            {showCommentBox && (
                <div className="absolute bottom-full right-0 mb-2 p-3 bg-white border border-gray-200 rounded-lg shadow-lg min-w-[300px] z-20">
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
}

const MessageArea = ({ messages, threadId, onSQLClick, openSQLModalForMsgId, onCloseSQLModal, onNewChat, isLoading }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    const containerRef = React.useRef<HTMLDivElement>(null);
    
    // State for feedback functionality
    const [feedbackState, setFeedbackState] = React.useState<{ [key: string]: { feedback: number | null; hasSubmitted: boolean } }>({});
    const [runIds, setRunIds] = React.useState<{ [key: string]: string }>({});
    
    // Auto-scroll to bottom when messages change or thread changes
    React.useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, threadId]);

    // Session and authentication
    const { data: session } = useSession();
    const userEmail = session?.user?.email || null;

    // Load feedback data for current thread and update runIds
    React.useEffect(() => {
        const loadFeedbackData = async () => {
            if (!threadId || !userEmail || !session?.id_token) return;
            
            try {
                const data = await authApiFetch<{run_ids: Array<{run_id: string, prompt: string, timestamp: string}>}>(`/chat/${threadId}/run-ids`, session.id_token);
                
                // Update runIds for legacy compatibility
                const runIdMap: { [key: string]: string } = {};
                const aiMessages = messages.filter(msg => !msg.isUser && !msg.isLoading);
                
                data.run_ids.forEach((item, index) => {
                    if (aiMessages[index]) {
                        runIdMap[aiMessages[index].id] = item.run_id;
                    }
                });
                
                setRunIds(runIdMap);
                
                // Update feedback state with proper structure
                const feedbackData: { [key: string]: { feedback: number | null; hasSubmitted: boolean } } = {};
                data.run_ids.forEach((item, index) => {
                    const messageKey = `msg_${index + 1}`;
                    feedbackData[messageKey] = {
                        feedback: null,
                        hasSubmitted: false
                    };
                });
                
                setFeedbackState(feedbackData);
            } catch (error) {
                console.error('[MessageArea] Error loading feedback data:', error);
            }
        };

        if (threadId && messages.length > 0) {
            loadFeedbackData();
        }
    }, [threadId, userEmail, session?.id_token, messages.length]);

    const handleFeedbackSubmit = async (messageId: string, feedback: number, comment?: string) => {
        // First try to get run_id from message meta, then from runIds state
        const message = messages.find(msg => msg.id === messageId);
        let runId = message?.meta?.run_id || runIds[messageId];
        
        if (!runId) {
            console.error('[MessageArea] No run_id found for message:', messageId);
            return;
        }

        try {
            // Get fresh session token
            const freshSession = await getSession();
            if (!freshSession?.id_token) {
                console.error('[MessageArea] No valid session for feedback submission');
                return;
            }

            await authApiFetch('/feedback', freshSession.id_token, {
                method: 'POST',
                body: JSON.stringify({
                    run_id: runId,
                    feedback: feedback,
                    comment: comment
                })
            });

            console.log('[MessageArea] Feedback submitted successfully for run_id:', runId);
            // Update feedback state to show submission
            setFeedbackState(prev => ({
                ...prev,
                [messageId]: { feedback, hasSubmitted: true }
            }));
        } catch (error) {
            console.error('[MessageArea] Error submitting feedback:', error);
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
                        disabled={isLoading}
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