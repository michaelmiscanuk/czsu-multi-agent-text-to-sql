"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession, getSession, signOut } from "next-auth/react";

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  selectionCode?: string | null;
  queriesAndResults?: [string, string][];
}

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
  const [chatSessions, setChatSessions] = useState<{id: string, title: string, messages: Message[]}[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [openSQLModalForMsgId, setOpenSQLModalForMsgId] = useState<number | null>(null);
  const [iteration, setIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(2); // default fallback
  // Track previous chatId and message count for scroll logic
  const prevChatIdRef = React.useRef<string | null>(null);
  const prevMsgCountRef = React.useRef<number>(1);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

  // Derive messages from active session
  const messages = React.useMemo(() => {
    const session = chatSessions.find(s => s.id === activeSessionId);
    return session ? session.messages : [{
      id: 1,
      content: 'Hi there, how can I help you?',
      isUser: false,
      type: 'message'
    }];
  }, [chatSessions, activeSessionId]);

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

  // Always fetch chat sessions from backend on mount if authenticated
  useEffect(() => {
    if (session?.id_token) {
      fetch(`${API_BASE}/chat-sessions`, {
        headers: { Authorization: `Bearer ${session.id_token}` }
      })
        .then(res => res.ok ? res.json() : Promise.reject(res))
        .then(data => {
          const backendSessions = Array.isArray(data) ? dedupeSessions(data.map(s => s.data)) : [];
          setChatSessions(backendSessions);
          // Restore last active chat from localStorage if possible
          const lastActiveId = localStorage.getItem('czsu-last-active-chat');
          if (lastActiveId && backendSessions.some(s => s.id === lastActiveId)) {
            setActiveSessionId(lastActiveId);
          } else if (backendSessions.length > 0) {
            setActiveSessionId(backendSessions[0].id);
          } else {
            setActiveSessionId(null);
          }
        })
        .catch(() => {
          setChatSessions([]);
          setActiveSessionId(null);
        });
    }
  }, [session?.id_token]);

  // Persist last active chat ID
  useEffect(() => {
    if (activeSessionId) {
      localStorage.setItem('czsu-last-active-chat', activeSessionId);
    }
  }, [activeSessionId]);

  // Save a single chat session to backend
  const saveChatSession = async (sessionObj: {id: string, title: string, messages: Message[]}) => {
    if (session?.id_token) {
      try {
        await fetch(`${API_BASE}/chat-sessions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${session.id_token}`
          },
          body: JSON.stringify(sessionObj)
        });
      } catch (e) {
        // ignore error
      }
    }
  };

  // Determine if we should auto-scroll
  useEffect(() => {
    const prevChatId = prevChatIdRef.current;
    const prevMsgCount = prevMsgCountRef.current;
    if (activeSessionId === prevChatId) {
      setShouldAutoScroll(messages.length > prevMsgCount);
    } else {
      setShouldAutoScroll(false);
    }
    prevChatIdRef.current = activeSessionId;
    prevMsgCountRef.current = messages.length;
  }, [activeSessionId, messages]);

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
    if (!currentMessage.trim()) return;
    // Always get a fresh session (token)
    const freshSession = await getSession();
    if (!freshSession?.id_token) {
      // Token missing or expired, sign out
      signOut();
      return;
    }
    let currentSession = chatSessions.find(s => s.id === activeSessionId);
    // If no session, create a new one and set as active
    if (!currentSession) {
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
        isLoading: true,
        startedAt: Date.now()
      };
      const newSessionMessages = [initialSystemMessage, userMessage, loadingMessage];
      const newSession = {
        id: newId,
        title: firstTitle || `Chat ${chatSessions.length + 1}`,
        messages: newSessionMessages
      };
      setChatSessions(dedupeSessions([newSession, ...chatSessions]));
      setActiveSessionId(newId);
      setCurrentMessage("");
      currentSession = newSession;
    }
    // If first user message, update title
    if (currentSession.messages.length === 1 && currentSession.messages[0].isUser === false) {
      const firstTitle = currentMessage.length > 30 ? currentMessage.slice(0, 30) + '…' : currentMessage;
      setChatSessions(prev => prev.map(s =>
        s.id === activeSessionId
          ? { ...s, title: firstTitle || s.title }
          : s
      ));
    }
    // Add user message and loading message to current session
    const newMessageId = currentSession.messages.length > 0 ? Math.max(...currentSession.messages.map((msg: any) => msg.id)) + 1 : 1;
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
      isLoading: true,
      startedAt: Date.now()
    };
    const updatedMessages = [...currentSession.messages, userMessage, loadingMessage];
    setChatSessions(prev => prev.map(s =>
      s.id === activeSessionId
        ? { ...s, messages: updatedMessages }
        : s
    ));
    // Save immediately after adding user and loading message
    const sessionToSave = { ...currentSession, messages: updatedMessages };
    await saveChatSession(sessionToSave);
    setCurrentMessage("");
    setIsLoading(true);
    try {
      const API_URL = `${API_BASE}/analyze`;
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${freshSession.id_token}`
      };
      const response = await fetch(API_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify({ prompt: currentMessage })
      });
      if (response.status === 401) {
        // Token expired or invalid, sign out
        signOut();
        setChatSessions(prev => prev.map(s =>
          s.id === activeSessionId
            ? { ...s, messages: s.messages.filter((msg: any) => !(hasIsLoading(msg) && msg.isLoading)) }
            : s
        ));
        setIsLoading(false);
        return;
      }
      if (!response.ok) {
        throw new Error('Server error');
      }
      const data = await response.json();
      setChatSessions(prev => prev.map(s =>
        s.id === activeSessionId
          ? { ...s, messages: s.messages.map((msg: any) =>
              hasIsLoading(msg) && msg.isLoading
                ? {
                    ...msg,
                    content: data.result || JSON.stringify(data),
                    isLoading: false,
                    selectionCode: data.selection_with_possible_answer || null,
                    queriesAndResults: Array.isArray(data.queries_and_results) ? data.queries_and_results : [],
                  }
                : msg
            ) }
          : s
      ));
      setIteration(data.iteration ?? 0);
      setMaxIterations(data.max_iterations ?? 2);
      // Save only the updated session
      const updatedSession = chatSessions.find(s => s.id === activeSessionId);
      if (updatedSession) await saveChatSession(updatedSession);
    } catch (error) {
      setChatSessions(prev => prev.map(s =>
        s.id === activeSessionId
          ? { ...s, messages: s.messages.filter((msg: any) => !(hasIsLoading(msg) && msg.isLoading)) }
          : s
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSQLButtonClick = (msgId: number) => {
    setOpenSQLModalForMsgId(msgId);
  };

  const handleCloseSQLModal = () => {
    setOpenSQLModalForMsgId(null);
  };

  // New chat handler
  const handleNewChat = async () => {
    const newId = uuidv4();
    const newSession = {
      id: newId,
      title: `Chat ${chatSessions.length + 1}`,
      messages: [{
        id: 1,
        content: 'Hi there, how can I help you?',
        isUser: false,
        type: 'message'
      }]
    };
    setChatSessions(dedupeSessions([newSession, ...chatSessions]));
    setActiveSessionId(newId);
    await saveChatSession(newSession);
  };

  // Clear all chats handler
  const handleClearChats = async () => {
    setChatSessions([]);
    setActiveSessionId(null);
    setCurrentMessage("");
    if (session?.id_token) {
      try {
        await fetch(`${API_BASE}/chat-sessions`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${session.id_token}` }
        });
      } catch (e) {
        // ignore error, just clear local state
      }
    }
  };

  // Delete a specific chat session
  const handleDeleteChat = async (chatId: string) => {
    setChatSessions(prev => prev.filter(s => s.id !== chatId));
    if (activeSessionId === chatId) {
      // If the deleted chat was active, switch to another or reset
      const remaining = chatSessions.filter(s => s.id !== chatId);
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id);
      } else {
        setActiveSessionId(null);
      }
    }
    if (session?.id_token) {
      try {
        await fetch(`${API_BASE}/chat-sessions/${chatId}`, {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${session.id_token}` }
        });
      } catch (e) {
        // ignore error, just update local state
      }
    }
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
          disabled={messages.length === 1 && messages[0].isUser === false}
          title={messages.length === 1 && messages[0].isUser === false ? 'You must send a message before starting a new chat.' : ''}
        >
          + New chat
        </button>
        <div className="flex-1 min-h-0 overflow-y-auto">
          {dedupeSessions(chatSessions).length === 0 ? (
            <div className="text-xs text-gray-400 mt-4">No chats yet</div>
          ) : (
            dedupeSessions(chatSessions).map(session => (
              <div key={session.id} className="relative group flex items-center">
                <button
                  className={`w-full text-left px-2 py-2 rounded transition-colors text-xs mb-1 ${activeSessionId === session.id ? 'bg-white font-bold border border-blue-200 shadow' : 'hover:bg-gray-200'}`}
                  onClick={() => {
                    setActiveSessionId(session.id);
                  }}
                >
                  {session.title}
                </button>
                <button
                  className="absolute right-1 top-1 text-gray-400 hover:text-red-500 text-lg font-bold opacity-0 group-hover:opacity-100 transition-opacity duration-150 px-1"
                  title="Delete chat"
                  onClick={e => {
                    e.stopPropagation();
                    handleDeleteChat(session.id);
                  }}
                  tabIndex={-1}
                >
                  ×
                </button>
              </div>
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
        {typeof window !== 'undefined' && (
          <pre style={{fontSize:10, color:'#aaa'}}>[DEBUG] showSQLModal: {String(openSQLModalForMsgId !== null)} | messages: {messages.length}</pre>
        )}
        <MessageArea
          messages={messages}
          chatId={activeSessionId}
          shouldAutoScroll={shouldAutoScroll}
          onSQLClick={handleSQLButtonClick}
          openSQLModalForMsgId={openSQLModalForMsgId}
          onCloseSQLModal={handleCloseSQLModal}
        />
        <div className="flex justify-center mb-2">
          <button
            className="px-4 py-1.5 bg-gradient-to-r from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 text-white rounded-full shadow text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleNewChat}
            disabled={isLoading || messages.length === 1 && messages[0].isUser === false}
            title={messages.length === 1 && messages[0].isUser === false ? 'You must send a message before starting a new chat.' : ''}
          >
            New chat
          </button>
        </div>
        <InputBar currentMessage={currentMessage} setCurrentMessage={setCurrentMessage} onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  );
} 