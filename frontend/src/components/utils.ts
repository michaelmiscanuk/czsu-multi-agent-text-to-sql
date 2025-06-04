// Utility to remove diacritics from a string (Czech and other languages)
export function removeDiacritics(str: string): string {
  return str.normalize('NFD').replace(/\p{Diacritic}/gu, '');
}

// chatDb.ts - Modern IndexedDB utility for chat sessions/messages
import { openDB, DBSchema } from 'idb';

export interface ChatSessionMeta {
  id: string;
  user: string; // user email
  title: string;
  createdAt: number;
  updatedAt: number;
}

export interface ChatMessage {
  id: string;
  sessionId: string;
  user: string;
  content: string;
  isUser: boolean;
  createdAt: number;
  error?: string;
  meta?: Record<string, any>;
  queriesAndResults?: [string, string][];
}

interface ChatDbSchema extends DBSchema {
  sessions: {
    key: string; // `${user}:${sessionId}`
    value: ChatSessionMeta;
    indexes: { 'by-user': string };
  };
  messages: {
    key: string; // `${user}:${sessionId}:${messageId}`
    value: ChatMessage;
    indexes: { 'by-session': string, 'by-user': string };
  };
}

export async function getChatDb() {
  return openDB<ChatDbSchema>('czsu-chat-modern', 3, {
    upgrade(db) {
      if (db.objectStoreNames.contains('sessions')) {
        db.deleteObjectStore('sessions');
      }
      const sessionStore = db.createObjectStore('sessions', { keyPath: 'id' });
      sessionStore.createIndex('by-user', 'user');

      if (db.objectStoreNames.contains('messages')) {
        db.deleteObjectStore('messages');
      }
      const msgStore = db.createObjectStore('messages', { keyPath: 'id' });
      msgStore.createIndex('by-session', 'sessionId');
      msgStore.createIndex('by-user', 'user');
    },
  });
}

// Session CRUD
export async function listSessions(user: string): Promise<ChatSessionMeta[]> {
  const db = await getChatDb();
  const sessions = await db.getAllFromIndex('sessions', 'by-user', user);
  console.log('[listSessions] For user:', user, 'Sessions:', JSON.stringify(sessions));
  return sessions.sort((a, b) => b.updatedAt - a.updatedAt);
}
export async function getChatSession(user: string, id: string): Promise<ChatSessionMeta | undefined> {
  const db = await getChatDb();
  return db.get('sessions', id);
}
export async function saveSession(meta: ChatSessionMeta) {
  const db = await getChatDb();
  await db.put('sessions', meta);
  console.log('[saveSession] Saved:', JSON.stringify(meta));
}
export async function deleteSession(user: string, id: string) {
  const db = await getChatDb();
  await db.delete('sessions', id);
  // Also delete all messages for this session
  const msgs = await listMessages(user, id);
  await Promise.all(msgs.map(m => db.delete('messages', m.id)));
}

// Message CRUD
export async function listMessages(user: string, sessionId: string): Promise<ChatMessage[]> {
  const db = await getChatDb();
  const all = await db.getAllFromIndex('messages', 'by-session', sessionId);
  return all.filter(m => m.user === user).sort((a, b) => a.createdAt - b.createdAt);
}
export async function saveMessage(msg: ChatMessage) {
  const db = await getChatDb();
  await db.put('messages', msg);
} 