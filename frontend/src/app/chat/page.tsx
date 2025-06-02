"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
}

const Modal: React.FC<{ open: boolean; onClose: () => void; children: React.ReactNode }> = ({ open, onClose, children }) => {
  React.useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full p-6 relative">
        <button
          className="absolute top-2 right-2 text-gray-400 hover:text-gray-700 text-2xl font-bold"
          onClick={onClose}
          title="Close"
        >
          ×
        </button>
        {children}
      </div>
    </div>
  );
};

const CHAT_STORAGE_KEY = 'czsu-chat-messages';
const LAST_SELECTION_CODE_KEY = 'czsu-chat-lastSelectionCode';
const LAST_QUERIES_RESULTS_KEY = 'czsu-chat-lastQueriesAndResults';
const INITIAL_MESSAGE = [
  {
    id: 1,
    content: 'Hi there, how can I help you?',
    isUser: false,
    type: 'message'
  }
];

export default function ChatPage() {
  const [messages, setMessages] = useState(INITIAL_MESSAGE);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [lastSelectionCode, setLastSelectionCode] = useState<string | null>(null);
  const [lastQueriesAndResults, setLastQueriesAndResults] = useState<[string, string][]>([]);
  const [showSQLModal, setShowSQLModal] = useState(false);

  // Only restore from localStorage on first mount
  const didRestoreRef = React.useRef(false);
  useEffect(() => {
    if (!didRestoreRef.current) {
      const saved = localStorage.getItem(CHAT_STORAGE_KEY);
      console.log('[ChatPage] Restoring chat from localStorage:', saved);
      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed) && parsed.length > 0) {
            setMessages(parsed);
            console.log('[ChatPage] Chat state after restore:', parsed);
          }
        } catch {}
      }
      // Restore lastSelectionCode
      const savedSel = localStorage.getItem(LAST_SELECTION_CODE_KEY);
      setLastSelectionCode(savedSel || null);
      // Restore lastQueriesAndResults
      const savedQR = localStorage.getItem(LAST_QUERIES_RESULTS_KEY);
      if (savedQR) {
        try {
          setLastQueriesAndResults(JSON.parse(savedQR));
        } catch { setLastQueriesAndResults([]); }
      }
      didRestoreRef.current = true;
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    console.log('[ChatPage] Persisting chat to localStorage:', messages);
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  // Persist lastSelectionCode
  useEffect(() => {
    if (lastSelectionCode) {
      localStorage.setItem(LAST_SELECTION_CODE_KEY, lastSelectionCode);
    } else {
      localStorage.removeItem(LAST_SELECTION_CODE_KEY);
    }
  }, [lastSelectionCode]);

  // Persist lastQueriesAndResults
  useEffect(() => {
    if (lastQueriesAndResults && lastQueriesAndResults.length > 0) {
      localStorage.setItem(LAST_QUERIES_RESULTS_KEY, JSON.stringify(lastQueriesAndResults));
    } else {
      localStorage.removeItem(LAST_QUERIES_RESULTS_KEY);
    }
  }, [lastQueriesAndResults]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLastQueriesAndResults([]);
    if (currentMessage.trim()) {
      const newMessageId = messages.length > 0 ? Math.max(...messages.map((msg: any) => msg.id)) + 1 : 1;
      setMessages((prev: any[]) => [
        ...prev,
        {
          id: newMessageId,
          content: currentMessage,
          isUser: true,
          type: 'message'
        }
      ]);
      const userInput = currentMessage;
      setCurrentMessage("");
      setIsLoading(true);
      try {
        // Add loading placeholder for AI response
        const aiResponseId = newMessageId + 1;
        setMessages((prev: any[]) => [
          ...prev,
          {
            id: aiResponseId,
            content: "",
            isUser: false,
            type: 'message',
            isLoading: true
          }
        ]);
        // Call your FastAPI backend
        const API_URL = process.env.NODE_ENV === 'development'
          ? 'http://localhost:8000/analyze'
          : '/analyze';
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ prompt: userInput })
        });
        if (!response.ok) {
          throw new Error('Server error');
        }
        const data = await response.json();
        setMessages((prev: any[]) =>
          prev.map((msg: any) =>
            msg.isLoading
              ? { ...msg, content: data.result || JSON.stringify(data), isLoading: false }
              : msg
          )
        );
        setLastSelectionCode(data.selection_with_possible_answer || null);
        if (Array.isArray(data.queries_and_results)) {
          setLastQueriesAndResults(data.queries_and_results);
        } else {
          setLastQueriesAndResults([]);
        }
      } catch (error) {
        setMessages((prev: any[]) =>
          prev.map((msg: any) =>
            msg.isLoading
              ? { ...msg, content: "Sorry, there was an error processing your request.", isLoading: false }
              : msg
          )
        );
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleSQLButtonClick = () => {
    setShowSQLModal(true);
  };

  const handleCloseSQLModal = () => {
    setShowSQLModal(false);
  };

  // New chat button handler
  const handleNewChat = () => {
    setMessages(INITIAL_MESSAGE);
    setLastSelectionCode(null);
    setLastQueriesAndResults([]);
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(INITIAL_MESSAGE));
  };

  return (
    <div className="w-full max-w-5xl bg-white flex flex-col rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh] p-8">
      <MessageArea messages={messages} />
      {lastSelectionCode &&
        <div className="mt-2 text-sm flex items-center space-x-4">
          <div>
            <span>Dataset used: </span>
            <Link
              href={`/data?table=${encodeURIComponent(lastSelectionCode)}`}
              className="text-blue-600 underline font-mono hover:text-blue-800"
            >
              {lastSelectionCode}
            </Link>
          </div>
          <button
            className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-xs font-semibold text-gray-700 border border-gray-300"
            onClick={handleSQLButtonClick}
          >
            SQL
          </button>
        </div>
      }
      <Modal open={showSQLModal} onClose={handleCloseSQLModal}>
        <h2 className="text-lg font-bold mb-4">SQL Commands & Results</h2>
        <div className="max-h-[60vh] overflow-y-auto pr-2">
          {(() => {
            // Deduplicate by SQL string
            const uniqueQueriesAndResults = Array.from(
              new Map(lastQueriesAndResults.map(([q, r]) => [q, [q, r]])).values()
            );
            if (uniqueQueriesAndResults.length === 0) {
              return <div className="text-gray-500">No SQL commands available.</div>;
            }
            return (
              <div className="space-y-6">
                {uniqueQueriesAndResults.map(([sql, result], idx) => (
                  <div key={idx} className="bg-gray-50 rounded border border-gray-200 p-0">
                    <div className="bg-gray-100 px-4 py-2 rounded-t text-xs font-semibold text-gray-700 border-b border-gray-200">SQL Command {idx + 1}</div>
                    <div className="p-3 font-mono text-xs whitespace-pre-line text-gray-900">
                      {sql.split('\n').map((line, i) => (
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
      <div className="flex justify-center mt-4 mb-2">
        <button
          className="px-3 py-1 bg-gradient-to-r from-teal-500 to-teal-400 hover:from-teal-600 hover:to-teal-500 text-white rounded-full text-xs font-medium transition-all duration-200"
          onClick={handleNewChat}
          disabled={isLoading}
        >
          New chat
        </button>
      </div>
      <InputBar currentMessage={currentMessage} setCurrentMessage={setCurrentMessage} onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
} 