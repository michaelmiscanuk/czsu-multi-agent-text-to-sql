import React from 'react';
import Modal from './Modal';
import Link from 'next/link';

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

interface MessageAreaProps {
    messages: any[];
    chatId: string | null;
    onSQLClick: (msgId: string) => void;
    openSQLModalForMsgId: string | null;
    onCloseSQLModal: () => void;
}

const MessageArea = ({ messages, chatId, onSQLClick, openSQLModalForMsgId, onCloseSQLModal }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    const containerRef = React.useRef<HTMLDivElement>(null);
    React.useEffect(() => {
        const container = containerRef.current;
        if (!container || !bottomRef.current) return;
        bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [messages, chatId]);
    return (
        <div ref={containerRef} className="flex-grow overflow-y-auto bg-[#FCFCF8] border-b border-gray-100" style={{ minHeight: 0 }} aria-live="polite" role="log">
            <div className="max-w-4xl mx-auto p-6 w-full">
                {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-6`}>
                        <div className="flex flex-col max-w-md w-full">
                            {/* Message Content */}
                            <div
                                className={`transition-all duration-200 rounded-3xl px-6 py-4 w-full select-text shadow-lg group
                                    ${message.isUser
                                        ? 'bg-gradient-to-br from-blue-500 via-blue-600 to-blue-700 text-white font-semibold rounded-br-3xl rounded-tr-3xl hover:shadow-xl'
                                        : 'bg-white border border-blue-100 text-gray-800 rounded-bl-3xl rounded-tl-3xl hover:shadow-xl'}
                                `}
                                style={{ fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)', fontSize: '0.97rem', lineHeight: 1.6, color: message.isUser ? '#fff' : undefined, wordBreak: 'break-word', whiteSpace: 'pre-line' }}
                            >
                                {message.isLoading && !message.content ? (
                                    <SimpleProgressBar key={message.id} messageId={message.id} startedAt={message.startedAt || Date.now()} />
                                ) : (
                                    message.content || (
                                        // Fallback if content is empty but not in loading state
                                        <span className="text-gray-400 text-xs italic">Waiting for response...</span>
                                    )
                                )}
                            </div>
                            {/* Dataset used and SQL button for AI answers */}
                            {!message.isUser && !message.isLoading && (message.selectionCode || message.meta?.datasetUrl || message.meta?.sql) && (
                                <div className="mt-3 flex items-center space-x-3" style={{ fontFamily: 'var(--font-inter, Inter, system-ui, sans-serif)' }}>
                                    {(message.selectionCode || message.meta?.datasetUrl) && (
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
                                    )}
                                    {message.meta?.sql && (
                                        <button
                                            className="px-4 py-1 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 text-white text-xs font-bold shadow hover:from-blue-500 hover:to-blue-700 border-0 transition-all duration-150"
                                            style={{ color: '#fff', textShadow: '0 1px 4px rgba(0,0,0,0.18)' }}
                                            onClick={() => onSQLClick(message.id)}
                                        >
                                            SQL
                                        </button>
                                    )}
                                    {/* SQL Modal for this message */}
                                    {openSQLModalForMsgId === message.id && (
                                        <Modal open={true} onClose={onCloseSQLModal}>
                                            <h2 className="text-lg font-bold mb-4">SQL Commands & Results</h2>
                                            <div className="max-h-[60vh] overflow-y-auto pr-2">
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
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default MessageArea;