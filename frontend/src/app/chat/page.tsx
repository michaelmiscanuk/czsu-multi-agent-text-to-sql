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
  deleteMessage,
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
  const sidebarRef = React.useRef<HTMLDivElement>(null);
  
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

  // Auto-scroll sidebar to top when entering chat menu
  useEffect(() => {
    if (sidebarRef.current) {
      sidebarRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

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
    
    // Create a loading message that will appear while waiting for response
    const loadingMsg: ChatMessage = {
      id: uuidv4(),
      threadId: threadId,
      user: userEmail,
      content: "",
      isUser: false,
      createdAt: Date.now(),
      isLoading: true,
      startedAt: Date.now(),
    };
    await saveMessage(loadingMsg);
    setMessages(await listMessages(userEmail, threadId));
    setCurrentMessage("");
    
    // Always update session title to first message if this is the first message in the session
    const threadMessages = await listMessages(userEmail, threadId);
    if (threadMessages.filter(m => m.isUser).length === 1) {
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
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Remove the loading message
      await deleteMessage(userEmail, loadingMsg.id);
      
      // Create the actual AI response message
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
      
      // Add warning if persistent memory is temporarily unavailable
      if (data.warning) {
        aiMsg.content = `⚠️ ${data.warning}\n\n${aiMsg.content}`;
      }
      
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
      console.error('Error sending message:', error);
      
      // Remove the loading message
      await deleteMessage(userEmail, loadingMsg.id);
      
      // Create an error message
      const errorMsg: ChatMessage = {
        id: uuidv4(),
        threadId: threadId,
        user: userEmail,
        content: `❌ Sorry, there was an error processing your request. Please try again.\n\nError details: ${error instanceof Error ? error.message : 'Unknown error'}`,
        isUser: false,
        createdAt: Date.now(),
        isError: true,
      };
      
      await saveMessage(errorMsg);
      setMessages(await listMessages(userEmail, threadId));
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
    <div className="chat-container flex w-full max-w-7xl mx-auto bg-white rounded-2xl shadow-2xl border border-gray-100 overflow-hidden">
      {/* Sidebar with its own scroll */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
          <span className="font-bold text-lg text-blue-700">Chats</span>
          <button
            className="px-3 py-1.5 rounded-full light-blue-theme text-sm font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleNewChat}
            title="New chat"
            disabled={isLoading || threads.some(s => !messages.length && s.id === activeThreadId)}
          >
            + New Chat
          </button>
        </div>
        
        {/* Sidebar Chat List with Scroll */}
        <div ref={sidebarRef} className="flex-1 overflow-y-auto overflow-x-hidden p-3 space-y-1 chat-scrollbar">
          {threads.length === 0 ? (
            <div className="text-sm text-gray-500 mt-8 text-center">No chats yet</div>
          ) : (
            threads.map(s => (
              <div key={s.id} className="group">
                {editingTitleId === s.id ? (
                  <input
                    className="w-full px-3 py-2 text-sm rounded-lg bg-white border border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
                    value={newTitle}
                    onChange={e => setNewTitle(e.target.value)}
                    onBlur={() => handleRename(s.id, newTitle)}
                    onKeyDown={e => { if (e.key === 'Enter') handleRename(s.id, newTitle); }}
                    autoFocus
                  />
                ) : (
                  <div className="flex items-center min-w-0">
                    <button
                      className={
                        `flex-1 text-left text-sm px-3 py-2 font-semibold rounded-lg transition-all duration-200 cursor-pointer min-w-0 ` +
                        (activeThreadId === s.id
                          ? 'font-extrabold light-blue-theme '
                          : 'text-[#181C3A]/80 hover:text-gray-300 hover:bg-gray-100 ')
                      }
                      style={{fontFamily: 'var(--font-inter)'}}
                      onClick={() => setActiveThreadId(s.id)}
                      onDoubleClick={() => { setEditingTitleId(s.id); setNewTitle(s.title); }}
                      title={s.title}
                    >
                      <span className="truncate block">{s.title}</span>
                    </button>
                    <button
                      className="flex-shrink-0 ml-1 text-gray-400 hover:text-red-500 text-lg font-bold px-2 py-1 rounded transition-colors"
                      title="Delete chat"
                      onClick={() => handleDelete(s.id)}
                    >
                      ×
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </aside>

      {/* Main Chat Container */}
      <div className="flex-1 flex flex-col bg-gradient-to-br from-white to-blue-50/30 relative">
        {/* Chat Messages Area with its own scroll */}
        <div className="flex-1 overflow-hidden">
          <MessageArea
            messages={messages}
            threadId={activeThreadId}
            onSQLClick={handleSQLButtonClick}
            openSQLModalForMsgId={openSQLModalForMsgId}
            onCloseSQLModal={handleCloseSQLModal}
            onNewChat={handleNewChat}
            isLoading={isLoading}
          />
        </div>
        
        {/* Stationary Input Field */}
        <div className="bg-white border-t border-gray-200 shadow-lg">
          <form onSubmit={handleSend} className="p-4 flex items-center gap-3 max-w-4xl mx-auto">
            <input
              ref={inputRef}
              type="text"
              placeholder="Type your message here..."
              value={currentMessage}
              onChange={e => setCurrentMessage(e.target.value)}
              className="flex-1 px-4 py-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 bg-gray-50 transition-all duration-200"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="px-6 py-3 light-blue-theme rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              disabled={isLoading || !currentMessage.trim()}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-gray-400 border-t-gray-600 rounded-full animate-spin"></div>
                  Sending...
                </span>
              ) : (
                <span>Send</span>
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
} 