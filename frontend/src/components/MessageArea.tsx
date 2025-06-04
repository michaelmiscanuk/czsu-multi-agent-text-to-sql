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
    shouldAutoScroll: boolean;
    onSQLClick: (msgId: number) => void;
    openSQLModalForMsgId: number | null;
    onCloseSQLModal: () => void;
}

const MessageArea = ({ messages, chatId, shouldAutoScroll, onSQLClick, openSQLModalForMsgId, onCloseSQLModal }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    const containerRef = React.useRef<HTMLDivElement>(null);
    React.useEffect(() => {
        if (!shouldAutoScroll) return;
        const container = containerRef.current;
        if (!container || !bottomRef.current) return;
        bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }, [messages, chatId, shouldAutoScroll]);
    return (
        <div ref={containerRef} className="flex-grow overflow-y-auto bg-[#FCFCF8] border-b border-gray-100" style={{ minHeight: 0 }} aria-live="polite" role="log">
            <div className="max-w-4xl mx-auto p-6">
                {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-5`}>
                        <div className="flex flex-col max-w-md w-full">
                            {/* Message Content */}
                            <div
                                className={`rounded-lg py-3 px-5 w-full ${message.isUser
                                    ? 'bg-gradient-to-br from-blue-500 to-blue-700 text-white rounded-br-none shadow-md'
                                    : 'bg-[#F3F3EE] text-gray-800 border border-gray-200 rounded-bl-none shadow-sm'
                                    }`}
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
                            {!message.isUser && message.selectionCode && !message.isLoading && (
                                <div className="mt-2 text-sm flex items-center space-x-4">
                                    <div>
                                        <span>Dataset used: </span>
                                        <Link
                                            href={`/data?table=${encodeURIComponent(message.selectionCode)}`}
                                            className="text-blue-600 underline font-mono hover:text-blue-800"
                                        >
                                            {message.selectionCode}
                                        </Link>
                                    </div>
                                    <button
                                        className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-xs font-semibold text-gray-700 border border-gray-300"
                                        onClick={() => onSQLClick(message.id)}
                                    >
                                        SQL
                                    </button>
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