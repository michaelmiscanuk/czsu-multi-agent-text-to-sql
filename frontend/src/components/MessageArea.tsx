import React from 'react';

const PROGRESS_DURATION = 20000; // 20 seconds

const SimpleProgressBar = ({ messageId }: { messageId: number }) => {
    const [progress, setProgress] = React.useState(0);
    const intervalRef = React.useRef<NodeJS.Timeout | null>(null);

    React.useEffect(() => {
        setProgress(0);
        const start = Date.now();
        intervalRef.current = setInterval(() => {
            const elapsed = Date.now() - start;
            const percent = Math.min(100, (elapsed / PROGRESS_DURATION) * 100);
            setProgress(percent);
            if (percent >= 100 && intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        }, 100);
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [messageId]);

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
}

const MessageArea = ({ messages }: MessageAreaProps) => {
    const bottomRef = React.useRef<HTMLDivElement>(null);
    React.useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'auto' });
        }
    }, [messages]);
    return (
        <div className="flex-grow overflow-y-auto bg-[#FCFCF8] border-b border-gray-100" style={{ minHeight: 0 }} aria-live="polite" role="log">
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
                                {message.isLoading ? (
                                    <SimpleProgressBar key={message.id} messageId={message.id} />
                                ) : (
                                    message.content || (
                                        // Fallback if content is empty but not in loading state
                                        <span className="text-gray-400 text-xs italic">Waiting for response...</span>
                                    )
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
};

export default MessageArea;