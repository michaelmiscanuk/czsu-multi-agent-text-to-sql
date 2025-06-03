"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession } from "next-auth/react";

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
const CHAT_SESSIONS_KEY = 'czsu-chat-sessions';
const INITIAL_MESSAGE = [
  {
    id: 1,
    content: 'Hi there, how can I help you?',
    isUser: false,
    type: 'message'
  }
];

export default function ChatPage() {
  const { data: session } = useSession();
  const [messages, setMessages] = useState(INITIAL_MESSAGE);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [lastSelectionCode, setLastSelectionCode] = useState<string | null>(null);
  const [lastQueriesAndResults, setLastQueriesAndResults] = useState<[string, string][]>([]);
  const [showSQLModal, setShowSQLModal] = useState(false);
  const [chatSessions, setChatSessions] = useState<{id: string, title: string, messages: Message[]}[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [iteration, setIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(2); // default fallback
  const didInitRef = React.useRef(false);
  const lastUserIdRef = React.useRef<string | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

  // Helper: deep compare sessions (by id and messages length)
  function sessionsAreEqual(a: any[], b: any[]) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i].id !== b[i].id) return false;
      if ((a[i].messages?.length || 0) !== (b[i].messages?.length || 0)) return false;
    }
    return true;
  }

  // Helper: deduplicate sessions by id (preserve full session object)
  function dedupeSessions<T extends {id: string}>(sessions: T[]): T[] {
    const seen = new Set();
    return sessions.filter(s => {
      if (seen.has(s.id)) return false;
      seen.add(s.id);
      return true;
    });
  }

  // Fetch chat sessions from backend only when user logs in or changes
  useEffect(() => {
    const userId = session?.user?.email || null;
    if (userId && lastUserIdRef.current !== userId && session?.id_token) {
      fetch(`${API_BASE}/chat-sessions`, {
        headers: { Authorization: `Bearer ${session.id_token}` }
      })
        .then(res => res.ok ? res.json() : Promise.reject(res))
        .then(data => {
          const backendSessions = Array.isArray(data) ? dedupeSessions(data.map(s => s.data)) : [];
          if (!sessionsAreEqual(backendSessions, chatSessions)) {
            setChatSessions(backendSessions);
            if (backendSessions.length > 0) {
              setActiveSessionId(backendSessions[0].id);
              setMessages(backendSessions[0].messages);
            } else {
              setActiveSessionId(null);
              setMessages(INITIAL_MESSAGE);
            }
          }
        })
        .catch(() => {
          setChatSessions([]);
          setActiveSessionId(null);
          setMessages(INITIAL_MESSAGE);
        });
      lastUserIdRef.current = userId;
      didInitRef.current = true;
    }
  }, [session?.id_token, session?.user?.email]);

  // Save chat sessions to backend whenever they change
  useEffect(() => {
    if (session?.id_token && chatSessions.length > 0) {
      chatSessions.forEach(sessionObj => {
        fetch(`${API_BASE}/chat-sessions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${session.id_token}`
          },
          body: JSON.stringify(sessionObj)
        });
      });
    }
  }, [chatSessions, session?.id_token]);

  // When activeSessionId changes, update messages
  useEffect(() => {
    if (activeSessionId) {
      const session = chatSessions.find(s => s.id === activeSessionId);
      if (session) setMessages(session.messages);
    }
  }, [activeSessionId, chatSessions]);

  // When messages change, update the current session in chatSessions
  useEffect(() => {
    if (activeSessionId) {
      setChatSessions(prev => prev.map(s => s.id === activeSessionId ? {...s, messages} : s));
    }
  }, [messages]);

  // Helper to check if current chat is empty (only system message)
  const isCurrentChatEmpty = messages.length === 1 && messages[0].isUser === false;

  // Debug: log session on mount and when it changes
  useEffect(() => {
    console.log('[ChatPage] Session:', JSON.stringify(session, null, 2));
  }, [session]);

  // Sync isLoading across tabs/windows
  useEffect(() => {
    function handleStorage(e: StorageEvent) {
      if (e.key === 'czsu-chat-isLoading') {
        setIsLoading(e.newValue === 'true');
      }
    }
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  useEffect(() => {
    if (isLoading) {
      localStorage.setItem('czsu-chat-isLoading', 'true');
    } else {
      localStorage.setItem('czsu-chat-isLoading', 'false');
    }
  }, [isLoading]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLastQueriesAndResults([]);
    if (currentMessage.trim()) {
      const currentSession = chatSessions.find(s => s.id === activeSessionId);
      const isCurrentSessionEmpty = !currentSession || (currentSession.messages.length === 1 && currentSession.messages[0].isUser === false);
      if (chatSessions.length === 0 || isCurrentSessionEmpty) {
        const newId = uuidv4();
        const firstTitle = currentMessage.length > 30 ? currentMessage.slice(0, 30) + '…' : currentMessage;
        const userMessage = {
          id: 2,
          content: currentMessage,
          isUser: true,
          type: 'message'
        };
        const initialSystemMessage = INITIAL_MESSAGE[0];
        const aiResponseId = 3;
        const loadingMessage = {
          id: aiResponseId,
          content: "",
          isUser: false,
          type: 'message',
          isLoading: true
        };
        const newSessionMessages = [initialSystemMessage, userMessage, loadingMessage];
        const newSession = {
          id: newId,
          title: firstTitle || `Chat ${chatSessions.length + 1}`,
          messages: newSessionMessages
        };
        setChatSessions(dedupeSessions([newSession, ...chatSessions]));
        setActiveSessionId(newId);
        setMessages(newSessionMessages);
        setCurrentMessage("");
        setIsLoading(true);
        try {
          const API_URL = `${API_BASE}/analyze`;
          const headers = {
            'Content-Type': 'application/json',
            ...(session?.id_token ? { 'Authorization': `Bearer ${session.id_token}` } : {}),
          };
          const response = await fetch(API_URL, {
            method: 'POST',
            headers,
            body: JSON.stringify({ prompt: currentMessage })
          });
          if (!response.ok) {
            throw new Error('Server error');
          }
          const data = await response.json();
          setMessages(prev =>
            prev.map(msg =>
              hasIsLoading(msg) && msg.isLoading
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
          setIteration(data.iteration ?? 0);
          setMaxIterations(data.max_iterations ?? 2);
        } catch (error) {
          setMessages(prev =>
            prev.map(msg =>
              hasIsLoading(msg) && msg.isLoading
                ? { ...msg, content: "Sorry, there was an error processing your request.", isLoading: false }
                : msg
            )
          );
        } finally {
          setIsLoading(false);
        }
        return;
      } else if (messages.length === 1 && messages[0].isUser === false) {
        setChatSessions(prev => prev.map(s =>
          s.id === activeSessionId
            ? { ...s, title: (currentMessage.length > 30 ? currentMessage.slice(0, 30) + '…' : currentMessage) || s.title }
            : s
        ));
      }
      const newMessageId = messages.length > 0 ? Math.max(...messages.map((msg: any) => msg.id)) + 1 : 1;
      const userMessage = {
        id: newMessageId,
        content: currentMessage,
        isUser: true,
        type: 'message'
      };
      const aiResponseId = newMessageId + 1;
      const loadingMessage = {
        id: aiResponseId,
        content: "",
        isUser: false,
        type: 'message',
        isLoading: true
      };
      setMessages((prev: any[]) => [
        ...prev,
        userMessage,
        loadingMessage
      ]);
      setCurrentMessage("");
      setIsLoading(true);
      try {
        const API_URL = `${API_BASE}/analyze`;
        const headers = {
          'Content-Type': 'application/json',
          ...(session?.id_token ? { 'Authorization': `Bearer ${session.id_token}` } : {}),
        };
        const response = await fetch(API_URL, {
          method: 'POST',
          headers,
          body: JSON.stringify({ prompt: currentMessage })
        });
        if (!response.ok) {
          throw new Error('Server error');
        }
        const data = await response.json();
        setMessages((prev: any[]) =>
          prev.map((msg: any) =>
            hasIsLoading(msg) && msg.isLoading
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
        setIteration(data.iteration ?? 0);
        setMaxIterations(data.max_iterations ?? 2);
      } catch (error) {
        setMessages((prev: any[]) =>
          prev.map((msg: any) =>
            hasIsLoading(msg) && msg.isLoading
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

  // New chat handler
  const handleNewChat = () => {
    if (messages.length === 1 && messages[0].isUser === false) return;
    const newId = uuidv4();
    if (chatSessions.some(s => s.id === newId)) return;
    const newSession = {
      id: newId,
      title: `Chat ${chatSessions.length + 1}`,
      messages: INITIAL_MESSAGE
    };
    setChatSessions(dedupeSessions([newSession, ...chatSessions]));
    setActiveSessionId(newId);
    setMessages(INITIAL_MESSAGE);
    setLastSelectionCode(null);
    setLastQueriesAndResults([]);
  };

  // Clear all chats handler
  const handleClearChats = () => {
    setChatSessions([]);
    setActiveSessionId(null);
    setMessages(INITIAL_MESSAGE);
    setLastSelectionCode(null);
    setLastQueriesAndResults([]);
    // Optionally: implement backend delete endpoint to clear all sessions
  };

  // Helper type guard
  function hasIsLoading(msg: any): msg is { isLoading: boolean } {
    return typeof msg === 'object' && msg !== null && 'isLoading' in msg;
  }

  // Sidebar UI
  return (
    <div className="flex w-full max-w-5xl bg-white rounded-2xl shadow-2xl border border-gray-100 overflow-hidden min-h-[70vh]">
      {/* Sidebar */}
      <aside className="w-48 bg-[#F9F9F5] border-r border-gray-200 shadow-sm flex flex-col p-2 text-gray-800">
        <button
          className="mb-2 px-3 py-1 bg-gradient-to-r from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 text-white rounded-full shadow text-xs font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleNewChat}
          disabled={isCurrentChatEmpty}
          title={isCurrentChatEmpty ? 'You must send a message before starting a new chat.' : ''}
        >
          + New chat
        </button>
        <div className="flex-1 min-h-0 overflow-y-auto">
          {dedupeSessions(chatSessions).length === 0 ? (
            <div className="text-xs text-gray-400 mt-4">No chats yet</div>
          ) : (
            dedupeSessions(chatSessions).map(session => (
              <button
                key={session.id}
                className={`w-full text-left px-2 py-2 rounded transition-colors text-xs mb-1 ${activeSessionId === session.id ? 'bg-white font-bold border border-blue-200 shadow' : 'hover:bg-gray-200'}`}
                onClick={() => setActiveSessionId(session.id)}
              >
                {session.title}
              </button>
            ))
          )}
        </div>
        <button
          className="mt-2 w-full px-3 py-1 bg-gradient-to-r from-gray-200 to-gray-100 hover:from-gray-300 hover:to-gray-200 text-gray-700 rounded-full shadow text-xs font-semibold transition-all duration-200 border border-gray-300"
          onClick={handleClearChats}
        >
          Clear Chats
        </button>
      </aside>
      {/* Main chat area */}
      <div className="flex-1 flex flex-col p-8">
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
        <div className="flex justify-center mb-2">
          <button
            className="px-4 py-1.5 bg-gradient-to-r from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 text-white rounded-full shadow text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleNewChat}
            disabled={isLoading || isCurrentChatEmpty}
            title={isCurrentChatEmpty ? 'You must send a message before starting a new chat.' : ''}
          >
            New chat
          </button>
        </div>
        <InputBar currentMessage={currentMessage} setCurrentMessage={setCurrentMessage} onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  );
} 