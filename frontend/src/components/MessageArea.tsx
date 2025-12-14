import React, { useEffect, useState } from 'react';
import { useSession, getSession } from "next-auth/react";
import Markdown from 'markdown-to-jsx';
import Modal from './Modal';
import Link from 'next/link';
import { ChatMessage, ChatThreadMeta, AnalyzeResponse, ChatThreadResponse } from '@/types';
import { API_CONFIG, authApiFetch } from '@/lib/api';
import { useSentiment } from '@/lib/useSentiment';
import { useChatCache } from '@/contexts/ChatCacheContext';

const PROGRESS_DURATION = 480000; // 8 minutes - matches backend analysis timeout
const SCROLL_LOCK_THRESHOLD_PX = 160;
type ScrollBehaviorType = 'auto' | 'smooth';

// Utility function to detect if text contains markdown
const containsMarkdown = (text: string): boolean => {
    if (!text) return false;
    
    // Check for common markdown patterns
    const markdownPatterns = [
        /\*\*[^*]+\*\*/,            // Bold text (**text**)
        /^\s*[-*+]\s+/m,            // Unordered lists (- * +)
        /^\s*\d+\.\s+/m,            // Ordered lists (1. 2. 3.)
        /^\s*\|.*\|.*$/m,           // Tables (|col1|col2|)
        /^#{1,6}\s+/m,              // Headers (# ## ### etc.)
        /`[^`]+`/,                  // Inline code (`code`)
        /```[\s\S]*?```/,           // Code blocks (```code```)
    ];
    
    return markdownPatterns.some(pattern => pattern.test(text));
};

// Component to render text with markdown support
interface MarkdownTextProps {
    content: string;
    className?: string;
    style?: React.CSSProperties;
}

const MarkdownText: React.FC<MarkdownTextProps> = ({ content, className, style }) => {
    const isMarkdown = containsMarkdown(content);
    
    // Base text styling that matches the message container
    const baseTextStyle = {
        ...style,
        fontFamily: 'var(--font-family)',
        fontSize: '0.97rem',
        lineHeight: 1.6,
        wordBreak: 'break-word' as const,
        color: '#374151' // text-gray-700 equivalent
    };

    if (isMarkdown) {
        const markdownOptions = {
            overrides: {
                // Customize paragraph spacing
                p: {
                    props: {
                        style: {
                            margin: '0 0 2px 0'
                        }
                    }
                },
                // Customize list spacing
                ul: {
                    props: {
                        style: {
                            margin: '0',
                            paddingLeft: '16px'
                        }
                    }
                },
                ol: {
                    props: {
                        style: {
                            margin: '0',
                            paddingLeft: '16px'
                        }
                    }
                },
                li: {
                    props: {
                        style: {
                            margin: '0 0 1px 0'
                        }
                    }
                },
                // Customize headers
                h1: {
                    props: {
                        style: {
                            fontSize: '1.25rem',
                            fontWeight: 'bold',
                            margin: '8px 0 4px 0'
                        }
                    }
                },
                h2: {
                    props: {
                        style: {
                            fontSize: '1.125rem',
                            fontWeight: 'bold',
                            margin: '8px 0 4px 0'
                        }
                    }
                },
                h3: {
                    props: {
                        style: {
                            fontSize: '1rem',
                            fontWeight: 'bold',
                            margin: '4px 0 4px 0'
                        }
                    }
                },
                h4: {
                    props: {
                        style: {
                            fontSize: '0.97rem',
                            fontWeight: 'bold',
                            margin: '4px 0 4px 0'
                        }
                    }
                },
                h5: {
                    props: {
                        style: {
                            fontSize: '0.875rem',
                            fontWeight: 'bold',
                            margin: '4px 0 4px 0'
                        }
                    }
                },
                h6: {
                    props: {
                        style: {
                            fontSize: '0.75rem',
                            fontWeight: 'bold',
                            margin: '4px 0 4px 0'
                        }
                    }
                },
                // Customize tables
                table: {
                    props: {
                        style: {
                            borderCollapse: 'collapse',
                            width: '100%',
                            border: '1px solid #d1d5db',
                            borderRadius: '6px',
                            fontSize: '0.875rem',
                            margin: '4px 0 8px 0'
                        }
                    }
                },
                thead: {
                    props: {
                        style: {
                            backgroundColor: '#f9fafb'
                        }
                    }
                },
                th: {
                    props: {
                        style: {
                            border: '1px solid #d1d5db',
                            padding: '8px 12px',
                            textAlign: 'left',
                            fontWeight: '600',
                            color: '#374151'
                        }
                    }
                },
                td: {
                    props: {
                        style: {
                            border: '1px solid #d1d5db',
                            padding: '8px 12px',
                            color: '#374151'
                        }
                    }
                },
                // Customize code blocks
                pre: {
                    props: {
                        style: {
                            backgroundColor: '#f3f4f6',
                            padding: '8px',
                            borderRadius: '6px',
                            margin: '4px 0',
                            overflow: 'auto',
                            fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace',
                            fontSize: '0.875rem'
                        }
                    }
                },
                code: {
                    props: {
                        style: {
                            backgroundColor: '#f3f4f6',
                            padding: '2px 4px',
                            borderRadius: '3px',
                            fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, monospace',
                            fontSize: '0.875rem'
                        }
                    }
                },
                // Customize block quotes
                blockquote: {
                    props: {
                        style: {
                            borderLeft: '4px solid #d1d5db',
                            paddingLeft: '12px',
                            margin: '4px 0',
                            fontStyle: 'italic',
                            color: '#6b7280'
                        }
                    }
                }
            }
        };

        return (
            <div className={className} style={baseTextStyle}>
                <Markdown options={markdownOptions}>{content}</Markdown>
            </div>
        );
    }
    
    // Fall back to regular text rendering with same styling
    return (
        <div 
            className={className} 
            style={{
                ...baseTextStyle,
                whiteSpace: 'pre-line'
            }}
        >
            {content}
        </div>
    );
};

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
        // Only proceed if we have a valid runId (don't fall back to messageId)
        if (!runId) {
            console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleFeedback - no valid runId, skipping feedback submission');
            return;
        }
        
        console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleFeedback - using runId:', runId, 'messageId:', messageId);
        
        // Save to separate localStorage for persistence
        saveFeedbackToLocalStorage(runId, feedback);
        
        // Update sentiment if we have a runId (new sentiment system)
        const sentiment = feedback === 1 ? true : false;
        onSentimentUpdate(runId, sentiment);
        
        // Call the original onFeedbackSubmit function (existing LangSmith feedback)
        onFeedbackSubmit(runId, feedback);
        setShowCommentBox(false);
        setComment('');
    };

    const handleCommentSubmit = () => {
        // Only proceed if we have a valid runId (don't fall back to messageId)
        if (!runId) {
            console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleCommentSubmit - no valid runId, skipping comment submission');
            return;
        }
        
        console.log('[FEEDBACK-DEBUG] FeedbackComponent.handleCommentSubmit - using runId:', runId, 'messageId:', messageId);
        const feedbackValue = messageFeedback.feedback !== null ? messageFeedback.feedback : 1;
        
        // Save to localStorage along with comment
        saveFeedbackToLocalStorage(runId, feedbackValue);
        
        // Call the comment submit function from MessageArea
        if (comment.trim()) {
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
                        disabled={!runId}
                        className={`p-1 rounded transition-colors ${
                            !runId 
                                ? 'text-gray-300 cursor-not-allowed' 
                                : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                        }`}
                        title={!runId ? "Feedback unavailable" : "Good response"}
                    >
                        👍
                    </button>
                    
                    {/* Thumbs down */}
                    <button
                        onClick={() => handleFeedback(0)}
                        disabled={!runId}
                        className={`p-1 rounded transition-colors ${
                            !runId 
                                ? 'text-gray-300 cursor-not-allowed' 
                                : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                        }`}
                        title={!runId ? "Feedback unavailable" : "Poor response"}
                    >
                        👎
                    </button>
                </>
            )}
            
            {/* Comment button with fixed positioning context */}
            <div className="relative">
                <button
                    ref={commentButtonRef}
                    onClick={() => runId && setShowCommentBox(!showCommentBox)}
                    disabled={!runId}
                    className={`p-1 rounded transition-colors ${
                        !runId
                            ? 'text-gray-300 cursor-not-allowed'
                            : showCommentBox 
                            ? 'text-blue-600 bg-blue-50' 
                            : hasProvidedComment
                            ? 'text-green-600 hover:text-green-700 hover:bg-green-50'
                            : 'text-gray-400 hover:text-blue-600 hover:bg-blue-50'
                    }`}
                    title={!runId ? "Feedback unavailable" : hasProvidedComment ? "Comment provided - click to edit" : "Add comment"}
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
    onFollowupPromptClick: (prompt: string) => void;
    onRerunPrompt?: (prompt: string) => void;
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

const MessageArea = ({ messages, threadId, onSQLClick, openSQLModalForMsgId, onCloseSQLModal, onPDFClick, openPDFModalForMsgId, onClosePDFModal, onFollowupPromptClick, onRerunPrompt, isLoading, isAnyLoading, threads, activeThreadId }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    const containerRef = React.useRef<HTMLDivElement>(null);
    const autoScrollRef = React.useRef(true);
    const streamingLockRef = React.useRef(false);
    const userOverrideRef = React.useRef(false);
    const lastScrollTopRef = React.useRef(0);
    const scrollTickingRef = React.useRef<number | null>(null);
    const [isPinnedToBottom, setIsPinnedToBottom] = React.useState(true);

    const scrollToBottom = React.useCallback((behavior: ScrollBehaviorType = 'smooth') => {
        if (!bottomRef.current) return;
        bottomRef.current.scrollIntoView({ behavior, block: 'end' });
    }, []);

    const handleJumpToLatest = React.useCallback(() => {
        userOverrideRef.current = false;
        autoScrollRef.current = true;
        setIsPinnedToBottom(true);
        scrollToBottom('smooth');
    }, [scrollToBottom]);
    
    // NEW: Debug messages changes
    React.useEffect(() => {
        console.log('[MessageArea] 🔄 Messages changed:', {
            threadId,
            messageCount: messages.length,
            loadingMessages: messages.filter(m => m.isLoading).length,
            completedMessages: messages.filter(m => m.final_answer && !m.isLoading).length
        });
    }, [messages, threadId]);

    React.useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        lastScrollTopRef.current = container.scrollTop;

        const updatePinnedState = (isScrollingDown = false) => {
            const distanceFromBottom = container.scrollHeight - (container.scrollTop + container.clientHeight);
            const nearBottom = distanceFromBottom <= SCROLL_LOCK_THRESHOLD_PX;

            if (userOverrideRef.current && nearBottom && isScrollingDown && distanceFromBottom <= 4) {
                userOverrideRef.current = false;
            }

            const pinned = nearBottom && !userOverrideRef.current && !streamingLockRef.current;
            setIsPinnedToBottom(nearBottom && !userOverrideRef.current);
            autoScrollRef.current = pinned;
        };

        const handleScroll = () => {
            if (scrollTickingRef.current !== null) {
                cancelAnimationFrame(scrollTickingRef.current);
            }

            scrollTickingRef.current = requestAnimationFrame(() => {
                const currentTop = container.scrollTop;
                const prevTop = lastScrollTopRef.current;
                const isScrollingUp = currentTop < prevTop;
                const isScrollingDown = currentTop > prevTop;

                if (isScrollingUp) {
                    userOverrideRef.current = true;
                    autoScrollRef.current = false;
                }

                lastScrollTopRef.current = currentTop;
                updatePinnedState(isScrollingDown);
            });
        };

        updatePinnedState();
        container.addEventListener('scroll', handleScroll, { passive: true });

        return () => {
            if (scrollTickingRef.current !== null) {
                cancelAnimationFrame(scrollTickingRef.current);
            }
            container.removeEventListener('scroll', handleScroll);
        };
    }, []);

    const lastMessageSignature = React.useMemo(() => {
        if (!messages.length) {
            return 'empty';
        }
        const lastMessage = messages[messages.length - 1];
        const followupSignature = lastMessage.followup_prompts?.join('||') ?? '';
        return `${lastMessage.id}:${lastMessage.final_answer?.length ?? 0}:${lastMessage.isLoading ? '1' : '0'}:${followupSignature}`;
    }, [messages]);
    
    // State for feedback functionality
    const [feedbackState, setFeedbackState] = React.useState<{ [runId: string]: { feedback: number | null; comment?: string } }>({});
    const [messageRunIds, setMessageRunIds] = React.useState<{[messageId: string]: string}>({});
    
    // NEW: Use ChatCacheContext for cached data access
    const { getRunIdsForThread, getSentimentsForThread, updateCachedSentiment } = useChatCache();
    
    // Use sentiments from cache instead of API calls
    const { sentiments, updateSentiment, loadSentiments, getSentimentForRunId } = useSentiment();
    
    // Track which run_ids have already sent feedback to LangSmith to prevent duplicates
    const [langsmithFeedbackSent, setLangsmithFeedbackSent] = React.useState<Set<string>>(new Set());
    
    // Auto-scroll when thread changes
    React.useEffect(() => {
        userOverrideRef.current = false;
        autoScrollRef.current = true;
        setIsPinnedToBottom(true);
        scrollToBottom('auto');
    }, [threadId, scrollToBottom]);

    // Detect if any message is currently streaming/loading
    const hasStreamingMessage = React.useMemo(() => {
        return messages.some((message) => message.isLoading);
    }, [messages]);

    // Lock auto-scroll while a response is streaming
    React.useEffect(() => {
        streamingLockRef.current = hasStreamingMessage;
        if (hasStreamingMessage) {
            autoScrollRef.current = false;
        } else if (!userOverrideRef.current && isPinnedToBottom) {
            autoScrollRef.current = true;
            scrollToBottom(messages.length <= 2 ? 'auto' : 'smooth');
        }
    }, [hasStreamingMessage, isPinnedToBottom, scrollToBottom, messages.length]);

    // Auto-scroll when latest message updates and user is pinned to bottom
    React.useEffect(() => {
        if (autoScrollRef.current) {
            scrollToBottom(messages.length <= 2 ? 'auto' : 'smooth');
        }
    }, [lastMessageSignature, messages.length, scrollToBottom]);

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
                
                // Get all messages that have final answers (AI responses) in order
                const aiMessages = messages.filter(message => message.final_answer);
                
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

    // CRITICAL: Load sentiments from cache into useSentiment hook state when thread changes
    React.useEffect(() => {
        if (!threadId) return;
        
        console.log('[SENTIMENT-DEBUG] Thread changed to:', threadId);
        console.log('[SENTIMENT-DEBUG] Loading sentiments from cache into useSentiment hook...');
        
        // CRITICAL: Call loadSentiments() to populate the useSentiment hook's local state
        // This loads the cached sentiments (from bulk loading) into the hook
        loadSentiments(threadId);
        
        console.log('[SENTIMENT-DEBUG] Sentiments loaded from cache for thread:', threadId);
    }, [threadId, loadSentiments]);

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
                            <p className="text-gray-500">Ask me about Czech Industry Statistical data and I'll help you analyze it!</p>
                        </div>
                    </div>
                ) : (
                    messages.map((message) => {
                        return (
                            <div key={message.id} className="message-container mb-6">
                                {/* Render user prompt if it exists */}
                                {message.prompt && (
                                    <div className="flex justify-end mb-4">
                                        <div className="flex items-center gap-3 max-w-2xl w-full">
                                            <div
                                                className="transition-all duration-200 rounded-2xl px-6 py-4 flex-1 select-text shadow-lg group light-blue-theme font-semibold hover:shadow-xl"
                                                style={{ 
                                                    fontFamily: 'var(--font-family)', 
                                                    fontSize: '0.97rem', 
                                                    lineHeight: 1.6, 
                                                    wordBreak: 'break-word', 
                                                    whiteSpace: 'pre-line' 
                                                }}
                                            >
                                                {message.prompt}
                                            </div>
                                            {/* Rerun button - only show when not loading */}
                                            {!isAnyLoading && onRerunPrompt && (
                                                <button
                                                    onClick={() => onRerunPrompt(message.prompt)}
                                                    className="flex-shrink-0 p-2 rounded-full bg-gray-100 hover:bg-gray-300 transition-colors duration-200"
                                                    title="Rerun this prompt"
                                                >
                                                    <svg className="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 24 24">
                                                        <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                                                    </svg>
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {/* Render assistant response if it exists or is loading */}
                                {(message.final_answer || message.isLoading) && (
                                    <div className="flex justify-start">
                                        <div className="flex flex-col max-w-2xl w-full">
                                            {/* Message Content with Copy Button */}
                                            <div className="relative">
                                                <div
                                                    className={`transition-all duration-200 rounded-2xl px-6 py-4 w-full select-text shadow-lg group
                                                        ${message.isError
                                                            ? 'bg-red-50 border border-red-200 text-red-800 hover:shadow-xl hover:border-red-300'
                                                            : 'bg-white border border-blue-100 text-gray-800 hover:shadow-xl hover:border-blue-200'}
                                                    `}
                                                    style={{ 
                                                        fontFamily: 'var(--font-family)', 
                                                        fontSize: '0.97rem', 
                                                        lineHeight: 1.6, 
                                                        wordBreak: 'break-word', 
                                                        whiteSpace: 'pre-line' 
                                                    }}
                                                    id={`message-content-${message.id}`}
                                                >
                                                    {/* Copy button - visible on hover or always on mobile */}
                                                    {!message.isLoading && message.final_answer && (
                                                        <button
                                                            onClick={async () => {
                                                                try {
                                                                    // Get the message content element
                                                                    const messageElement = document.getElementById(`message-content-${message.id}`);
                                                                    if (!messageElement) return;
                                                                    
                                                                    // Create a temporary container to convert markdown to HTML
                                                                    const tempDiv = document.createElement('div');
                                                                    tempDiv.innerHTML = messageElement.innerHTML;
                                                                    
                                                                    // Remove the copy button from the clone
                                                                    const copyButton = tempDiv.querySelector('button');
                                                                    if (copyButton) {
                                                                        copyButton.remove();
                                                                    }
                                                                    
                                                                    // Get HTML content for rich text copy
                                                                    const htmlContent = tempDiv.innerHTML;
                                                                    
                                                                    // Get plain text as fallback
                                                                    const plainText = message.final_answer;
                                                                    
                                                                    // Copy both HTML and plain text to clipboard
                                                                    // This allows pasting formatted text into Word and plain text into text editors
                                                                    const clipboardItem = new ClipboardItem({
                                                                        'text/html': new Blob([htmlContent], { type: 'text/html' }),
                                                                        'text/plain': new Blob([plainText], { type: 'text/plain' })
                                                                    });
                                                                    
                                                                    await navigator.clipboard.write([clipboardItem]);
                                                                    
                                                                    // Visual feedback
                                                                    const button = document.getElementById(`copy-button-${message.id}`);
                                                                    if (button) {
                                                                        const svg = button.querySelector('svg');
                                                                        const path = svg?.querySelector('path');
                                                                        if (path) {
                                                                            // Store original path
                                                                            const originalPath = path.getAttribute('d');
                                                                            const originalColor = svg?.getAttribute('stroke');
                                                                            
                                                                            // Replace with checkmark path
                                                                            path.setAttribute('d', 'M5 13l4 4L19 7');
                                                                            svg?.setAttribute('stroke', '#10b981');
                                                                            
                                                                            setTimeout(() => {
                                                                                // Restore original copy icon
                                                                                if (originalPath) {
                                                                                    path.setAttribute('d', originalPath);
                                                                                }
                                                                                if (originalColor) {
                                                                                    svg?.setAttribute('stroke', originalColor);
                                                                                } else {
                                                                                    svg?.setAttribute('stroke', 'currentColor');
                                                                                }
                                                                            }, 2000);
                                                                        }
                                                                    }
                                                                } catch (err) {
                                                                    console.error('Failed to copy text:', err);
                                                                }
                                                            }}
                                                            id={`copy-button-${message.id}`}
                                                            className="absolute top-3 right-3 p-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-gray-800 transition-all duration-150 opacity-0 group-hover:opacity-100 focus:opacity-100"
                                                            title="Copy formatted text"
                                                        >
                                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                                            </svg>
                                                        </button>
                                                    )}
                                                    
                                                    {(() => {
                                                        if (message.isLoading && !message.final_answer) {
                                                            return (
                                                                <div className="flex items-center space-x-3">
                                                                    <div className="w-5 h-5 border-2 border-blue-300 border-t-blue-600 rounded-full animate-spin"></div>
                                                                    <span className="text-gray-600">Analyzing your request...</span>
                                                                </div>
                                                            );
                                                        } else if (message.final_answer) {
                                                            return (
                                                                <MarkdownText 
                                                                    content={message.final_answer}
                                                                    style={{ 
                                                                        fontFamily: 'var(--font-family)', 
                                                                        fontSize: '0.97rem', 
                                                                        lineHeight: 1.6, 
                                                                        wordBreak: 'break-word' 
                                                                    }}
                                                                />
                                                            );
                                                        } else {
                                                            return (
                                                                <span className="text-gray-400 text-xs italic">Waiting for response...</span>
                                                            );
                                                        }
                                                    })()}
                                                </div>
                                            </div>

                                            {/* Dataset used and SQL button for AI answers */}
                                            {!message.isLoading && (message.datasets_used?.length || message.sql_query || message.top_chunks?.length) && (
                                        <div className="mt-3 flex items-center flex-wrap gap-3" style={{ fontFamily: 'var(--font-family)' }}>
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

                                            {/* Feedback component - moves to right when wrapping */}
                                            {threadId && (
                                                <div className="ml-auto">
                                                    <FeedbackComponent
                                                        messageId={message.id}
                                                        runId={message.run_id || messageRunIds[message.id]}
                                                        threadId={threadId}
                                                        onFeedbackSubmit={handleFeedbackSubmit}
                                                        onCommentSubmit={handleCommentSubmit}
                                                        feedbackState={feedbackState}
                                                        currentSentiment={getSentimentForRunId(message.run_id || messageRunIds[message.id] || '')}
                                                        onSentimentUpdate={updateSentiment}
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    )}

                                            {/* Show feedback even when no datasets/SQL */}
                                            {!message.isLoading && threadId && !(message.datasets_used?.length || message.sql_query || message.top_chunks?.length) && (
                                        <div className="mt-3 flex justify-end">
                                            <FeedbackComponent
                                                messageId={message.id}
                                                runId={message.run_id || messageRunIds[message.id]}
                                                threadId={threadId}
                                                onFeedbackSubmit={handleFeedbackSubmit}
                                                onCommentSubmit={handleCommentSubmit}
                                                feedbackState={feedbackState}
                                                currentSentiment={getSentimentForRunId(message.run_id || messageRunIds[message.id] || '')}
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
                                )}
                            </div>
                        );
                    })
                )}
                
                {/* Follow-up Prompts at bottom - only show after the last message */}
                {(() => {
                    // Get the latest message with follow-up prompts
                    const latestMessageWithPrompts = messages
                        .slice()
                        .reverse()
                        .find(msg => msg.followup_prompts && msg.followup_prompts.length > 0);
                    
                    const followupPrompts = latestMessageWithPrompts?.followup_prompts || [];
                    
                    return followupPrompts.length > 0 && !isAnyLoading ? (
                        <div className="pt-0 pb-0 flex flex-wrap gap-2 items-center">
                            {followupPrompts.map((prompt: string, index: number) => (
                                <button
                                    key={index}
                                    onClick={() => onFollowupPromptClick(prompt)}
                                    disabled={isAnyLoading}
                                    className="inline-block px-3 py-1 bg-white hover:bg-gray-50 border border-gray-200 rounded-full text-sm text-gray-700 hover:text-gray-900 transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                                    style={{
                                        fontFamily: 'var(--font-family)',
                                    }}
                                >
                                    {prompt}
                                </button>
                            ))}
                        </div>
                    ) : null;
                })()}

                {!isPinnedToBottom && (
                    <div className="sticky bottom-4 flex justify-center pointer-events-none mt-4">
                        <button
                            type="button"
                            onClick={handleJumpToLatest}
                            className="pointer-events-auto px-5 py-2 rounded-full light-blue-theme shadow-lg text-sm font-semibold flex items-center gap-2"
                            aria-label="Jump to latest message"
                        >
                            Jump to latest
                            <span aria-hidden="true">↓</span>
                        </button>
                    </div>
                )}
                
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default MessageArea;