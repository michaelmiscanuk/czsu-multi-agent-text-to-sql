import React from 'react';

const PremiumTypingAnimation = () => {
    return (
        <div className="flex items-center">
            <div className="flex items-center space-x-1.5">
                <div className="w-1.5 h-1.5 bg-gray-400/70 rounded-full animate-pulse"
                    style={{ animationDuration: "1s", animationDelay: "0ms" }}></div>
                <div className="w-1.5 h-1.5 bg-gray-400/70 rounded-full animate-pulse"
                    style={{ animationDuration: "1s", animationDelay: "300ms" }}></div>
                <div className="w-1.5 h-1.5 bg-gray-400/70 rounded-full animate-pulse"
                    style={{ animationDuration: "1s", animationDelay: "600ms" }}></div>
            </div>
        </div>
    );
};

const MessageArea = ({ messages }: { messages: any[] }) => {
    return (
        <div className="flex-grow overflow-y-auto bg-[#FCFCF8] border-b border-gray-100" style={{ minHeight: 0 }}>
            <div className="max-w-4xl mx-auto p-6">
                {messages.map((message) => (
                    <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-5`}>
                        <div className="flex flex-col max-w-md">
                            {/* Message Content */}
                            <div
                                className={`rounded-lg py-3 px-5 ${message.isUser
                                    ? 'bg-gradient-to-br from-[#5E507F] to-[#4A3F71] text-white rounded-br-none shadow-md'
                                    : 'bg-[#F3F3EE] text-gray-800 border border-gray-200 rounded-bl-none shadow-sm'
                                    }`}
                            >
                                {message.isLoading ? (
                                    <PremiumTypingAnimation />
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
            </div>
        </div>
    );
};

export default MessageArea;