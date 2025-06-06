"use client";
import InputBar from '@/components/InputBar';
import MessageArea from '@/components/MessageArea';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { v4 as uuidv4 } from 'uuid';
import { useSession, getSession, signOut } from "next-auth/react";
import {
  listThreads,
  getChatThread,
  saveThread,
  deleteThread,
  listMessages,
  saveMessage,
  ChatThreadMeta,
  ChatMessage
} from '@/components/utils';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  type: string;
  isLoading?: boolean;
  selectionCode?: string | null;
  queriesAndResults?: [string, string][];
  meta?: {
    datasetUrl?: string;
    datasetCodes?: string[];  // Array of dataset codes actually used in queries
    sql?: string;
  };
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
  const { data: session, update } = useSession();
  const userEmail = session?.user?.email || null;
  if (!userEmail) {
    return <div>Loading...</div>;
  }
  // const userEmail = "test3@test.com"
  const [threads, setThreads] = useState<ChatThreadMeta[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [editingTitleId, setEditingTitleId] = useState<string | null>(null);
  const [newTitle, setNewTitle] = useState("");
  const [openSQLModalForMsgId, setOpenSQLModalForMsgId] = useState<string | null>(null);
  const [iteration, setIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(2); // default fallback
  // Track previous chatId and message count for scroll logic
  const prevChatIdRef = React.useRef<string | null>(null);
  const prevMsgCountRef = React.useRef<number>(1);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || '';

  // Load sessions on mount/user change
  useEffect(() => {
    if (!userEmail) return;
    listThreads(userEmail).then(setThreads);
  }, [userEmail]);

  // Load messages for active session
  useEffect(() => {
    if (!userEmail || !activeThreadId) {
      setMessages([]);
      return;
    }
    listMessages(userEmail, activeThreadId).then(setMessages);
  }, [userEmail, activeThreadId]);

  // Remember last active chat in localStorage
  useEffect(() => {
    if (activeThreadId) {
      localStorage.setItem('czsu-last-active-chat', activeThreadId);
    }
  }, [activeThreadId]);

  // Restore last active chat on mount
  useEffect(() => {
    if (!userEmail) return;
    const lastActive = localStorage.getItem('czsu-last-active-chat');
    if (lastActive) {
      setActiveThreadId(lastActive);
    } else if (threads.length > 0 && !activeThreadId) {
      setActiveThreadId(threads[0].id);
    }
    // eslint-disable-next-line
  }, [userEmail, threads.length]);

  // Robust auto-create: Only after loading sessions from storage
  useEffect(() => {
    if (!userEmail) return;
    (async () => {
      const loadedThreads = await listThreads(userEmail);
      setThreads(loadedThreads);
      if (loadedThreads.length === 0) {
        const id = uuidv4();
        const now = Date.now();
        const meta: ChatThreadMeta = {
          id,
          user: userEmail,
          title: 'New Chat',
          createdAt: now,
          updatedAt: now,
        };
        await saveThread(meta);
        setThreads([meta]);
        setActiveThreadId(id);
        setTimeout(() => inputRef.current?.focus(), 0);
      }
    })();
  }, [userEmail]);

  // New chat
  const handleNewChat = async () => {
    if (!userEmail) return;
    const id = uuidv4();
    const now = Date.now();
    const meta: ChatThreadMeta = {
      id,
      user: userEmail,
      title: 'New Chat',
      createdAt: now,
      updatedAt: now,
    };
    await saveThread(meta);
    setThreads(await listThreads(userEmail));
    setActiveThreadId(id);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  // Rename chat
  const handleRename = async (id: string, title: string) => {
    if (!userEmail) return;
    const meta = await getChatThread(userEmail, id);
    if (meta) {
      meta.title = title;
      meta.updatedAt = Date.now();
      await saveThread(meta);
      setThreads(await listThreads(userEmail));
      console.log('[ChatPage] Sidebar sessions after title update:', await listThreads(userEmail));
      setEditingTitleId(null);
    }
  };

  // Delete chat
  const handleDelete = async (id: string) => {
    if (!userEmail) return;
    await deleteThread(userEmail, id);
    const updated = await listThreads(userEmail);
    setThreads(updated);
    if (activeThreadId === id) {
      setActiveThreadId(updated[0]?.id || null);
    }
  };

  // Send message
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userEmail || !currentMessage.trim()) return;
    setIsLoading(true);
    let threadId = activeThreadId;
    let isNewThread = false;
    // If no session exists, create a new one
    if (!threadId) {
      threadId = uuidv4();
      const now = Date.now();
      const meta: ChatThreadMeta = {
        id: threadId,
        user: userEmail,
        title: currentMessage.slice(0, 30), // Use first message as title, max 30 chars, no ellipsis
        createdAt: now,
        updatedAt: now,
      };
      await saveThread(meta);
      setThreads(await listThreads(userEmail));
      console.log('[ChatPage] Sidebar sessions after save:', await listThreads(userEmail));
      setActiveThreadId(threadId);
      isNewThread = true;
    }
    const msg: ChatMessage = {
      id: uuidv4(),
      threadId: threadId,
      user: userEmail,
      content: currentMessage,
      isUser: true,
      createdAt: Date.now(),
    };
    await saveMessage(msg);
    setMessages(await listMessages(userEmail, threadId));
    setCurrentMessage("");
    // Always update session title to first message if this is the first message in the session
    const threadMessages = await listMessages(userEmail, threadId);
    if (threadMessages.length === 1) {
      const meta = await getChatThread(userEmail, threadId);
      if (meta) {
        meta.title = msg.content.slice(0, 30);
        meta.updatedAt = Date.now();
        await saveThread(meta);
        setThreads(await listThreads(userEmail));
        console.log('[ChatPage] Sidebar sessions after title update:', await listThreads(userEmail));
      }
    }
    // Call backend for AI response
    try {
      let freshSession = await getSession();
      if (!freshSession?.id_token) {
        signOut();
        setIsLoading(false);
        return;
      }
      const API_URL = `${API_BASE}/analyze`;
      const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${freshSession.id_token}`
      };
      let response = await fetch(API_URL, {
        method: 'POST',
        headers,
        body: JSON.stringify({ prompt: msg.content, thread_id: threadId })
      });
      // If 401, try to force session refresh and retry once
      if (response.status === 401) {
        // Force session update (refresh token)
        const refreshedSession = await update();
        if (!refreshedSession?.id_token) {
          signOut();
          setIsLoading(false);
          return;
        }
        const retryHeaders = {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${refreshedSession.id_token}`
        };
        response = await fetch(API_URL, {
          method: 'POST',
          headers: retryHeaders,
          body: JSON.stringify({ prompt: msg.content, thread_id: threadId })
        });
        if (response.status === 401) {
          signOut();
          setIsLoading(false);
          return;
        }
      }
      if (!response.ok) {
        throw new Error('Server error');
      }
      const data = await response.json();
      const aiMsg: ChatMessage = {
        id: uuidv4(),
        threadId: threadId,
        user: userEmail,
        content: data.result || JSON.stringify(data),
        isUser: false,
        createdAt: Date.now(),
        queriesAndResults: data.queries_and_results,
        meta: {
          datasetUrl: data.datasetUrl,
          datasetCodes: data.top_selection_codes || [],  // Extract used selection codes
          sql: data.sql
        }
      };
      await saveMessage(aiMsg);
      setMessages(await listMessages(userEmail, threadId));
      // Update session updatedAt
      const meta = await getChatThread(userEmail, threadId);
      if (meta) {
        meta.updatedAt = Date.now();
        // If this was a new session, update the title to the first message
        if (isNewThread) {
          meta.title = msg.content.slice(0, 30);
        }
        await saveThread(meta);
        setThreads(await listThreads(userEmail));
        console.log('[ChatPage] Sidebar sessions after title update:', await listThreads(userEmail));
      }
    } catch (error) {
      // Optionally show error message
    } finally {
      setIsLoading(false);
    }
  };

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

  const handleSQLButtonClick = (msgId: string) => {
    setOpenSQLModalForMsgId(msgId);
  };

  const handleCloseSQLModal = () => {
    setOpenSQLModalForMsgId(null);
  };

  // UI
  return (
    <div className="flex flex-1 min-h-0 w-full max-w-6xl mx-auto bg-white rounded-2xl shadow-2xl border border-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 bg-gradient-to-b from-blue-50 to-white border-r border-gray-200 flex flex-col p-3">
        <div className="flex items-center mb-4">
          <span className="font-bold text-lg text-blue-700">Chats</span>
          <button
            className="ml-auto px-2 py-1 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 text-white text-xs font-bold shadow hover:from-blue-500 hover:to-blue-700 border-0 transition-all duration-150"
            style={{ color: '#fff', textShadow: '0 1px 4px rgba(0,0,0,0.18)' }}
            onClick={handleNewChat}
            title="New chat"
            disabled={isLoading || threads.some(s => !messages.length && s.id === activeThreadId)}
          >
            +
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {threads.length === 0 ? (
            <div className="text-xs text-gray-400 mt-4">No chats yet</div>
          ) : (
            threads.map(s => (
              <div key={s.id} className={`group flex items-center mb-1 rounded ${activeThreadId === s.id ? 'bg-blue-100' : 'hover:bg-gray-100'}`}>
                {editingTitleId === s.id ? (
                  <input
                    className="flex-1 px-2 py-1 text-xs rounded bg-white border border-blue-300 focus:outline-none"
                    value={newTitle}
                    onChange={e => setNewTitle(e.target.value)}
                    onBlur={() => handleRename(s.id, newTitle)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRename(s.id, newTitle); }}
                    autoFocus
                  />
                ) : (
                  <button
                    className={`flex-1 text-left px-2 py-2 text-xs font-medium truncate ${activeThreadId === s.id ? 'text-blue-800 font-bold' : 'text-gray-700'}`}
                    onClick={() => setActiveThreadId(s.id)}
                    onDoubleClick={() => { setEditingTitleId(s.id); setNewTitle(s.title); }}
                  >
                    {s.title}
                  </button>
                )}
                <button
                  className="ml-1 text-gray-400 hover:text-red-500 text-lg font-bold px-1"
                  title="Delete chat"
                  onClick={() => handleDelete(s.id)}
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>
      </aside>
      {/* Main chat area */}
      <div className="flex-1 flex flex-col bg-gradient-to-br from-white to-blue-50">
        <div className="flex-1 overflow-y-auto p-8">
          <MessageArea
            messages={messages}
            threadId={activeThreadId}
            onSQLClick={handleSQLButtonClick}
            openSQLModalForMsgId={openSQLModalForMsgId}
            onCloseSQLModal={handleCloseSQLModal}
          />
        </div>
        {/* Duplicated New Chat button at the bottom */}
        <div className="flex justify-center pb-2">
          <button
            className="px-4 py-1 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 text-white text-xs font-bold shadow hover:from-blue-500 hover:to-blue-700 border-0 transition-all duration-150"
            style={{ color: '#fff', textShadow: '0 1px 4px rgba(0,0,0,0.18)' }}
            onClick={handleNewChat}
            title="New chat"
            disabled={isLoading || threads.some(s => !messages.length && s.id === activeThreadId)}
          >
            +
          </button>
        </div>
        {/* Input bar */}
        <form onSubmit={handleSend} className="p-4 bg-white border-t border-gray-100 flex items-center">
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a message…"
            value={currentMessage}
            onChange={e => setCurrentMessage(e.target.value)}
            className="flex-1 px-4 py-2 rounded-full border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-200 text-gray-700 bg-blue-50"
            disabled={isLoading}
          />
          <button
            type="submit"
            className="ml-2 px-5 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-full font-semibold shadow disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isLoading || !currentMessage.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
} 