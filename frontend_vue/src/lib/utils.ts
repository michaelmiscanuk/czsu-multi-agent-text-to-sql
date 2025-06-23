import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { ChatMessage as SharedChatMessage, ChatThreadMeta as SharedChatThreadMeta } from '@/types';

// Utility to remove diacritics from a string (Czech and other languages)
export function removeDiacritics(str: string): string {
  return str.normalize('NFD').replace(/\p{Diacritic}/gu, '');
}

// Local interface for IndexedDB (extends the base ChatThreadMeta)
export interface LocalChatThreadMeta extends Omit<SharedChatThreadMeta, 'thread_id' | 'latest_timestamp' | 'run_count' | 'full_prompt'> {
  id: string;
  user: string; // user email
  createdAt: number;
  updatedAt: number;
}

// Use the shared ChatMessage interface
export type ChatMessage = SharedChatMessage;

interface ChatDbSchema extends DBSchema {
  threads: {
    key: string; // `${user}:${threadId}`
    value: LocalChatThreadMeta;
    indexes: { 'by-user': string };
  };
  messages: {
    key: string; // `${user}:${threadId}:${messageId}`
    value: ChatMessage;
    indexes: { 'by-thread': string, 'by-user': string };
  };
}

let dbPromise: Promise<IDBPDatabase<ChatDbSchema>> | null = null;

export async function getChatDb(): Promise<IDBPDatabase<ChatDbSchema>> {
  if (!dbPromise) {
    dbPromise = openDB<ChatDbSchema>('czsu-chat-modern', 4, {
      upgrade(db) {
        if (db.objectStoreNames.contains('threads')) {
          db.deleteObjectStore('threads');
        }
        const threadStore = db.createObjectStore('threads', { keyPath: 'id' });
        threadStore.createIndex('by-user', 'user');

        if (db.objectStoreNames.contains('messages')) {
          db.deleteObjectStore('messages');
        }
        const msgStore = db.createObjectStore('messages', { keyPath: 'id' });
        msgStore.createIndex('by-thread', 'threadId');
        msgStore.createIndex('by-user', 'user');
      },
    });
  }
  return dbPromise;
}

// Session CRUD
export async function listThreads(user: string): Promise<LocalChatThreadMeta[]> {
  const db = await getChatDb();
  const threads = await db.getAllFromIndex('threads', 'by-user', user);
  console.log('[listThreads] For user:', user, 'Threads:', JSON.stringify(threads));
  return threads.sort((a, b) => b.updatedAt - a.updatedAt);
}

export async function getChatThread(user: string, id: string): Promise<LocalChatThreadMeta | undefined> {
  const db = await getChatDb();
  return db.get('threads', id);
}

export async function saveThread(meta: LocalChatThreadMeta): Promise<void> {
  const db = await getChatDb();
  await db.put('threads', meta);
  console.log('[saveThread] Saved:', JSON.stringify(meta));
}

export async function deleteThread(user: string, id: string): Promise<void> {
  const db = await getChatDb();
  await db.delete('threads', id);
  // Also delete all messages for this thread
  const msgs = await listMessages(user, id);
  await Promise.all(msgs.map(m => db.delete('messages', m.id)));
}

// Message CRUD
export async function listMessages(user: string, threadId: string): Promise<ChatMessage[]> {
  const db = await getChatDb();
  const all = await db.getAllFromIndex('messages', 'by-thread', threadId);
  return all.filter(m => m.user === user).sort((a, b) => a.createdAt - b.createdAt);
}

export async function saveMessage(msg: ChatMessage): Promise<void> {
  const db = await getChatDb();
  await db.put('messages', msg);
}

export async function deleteMessage(user: string, messageId: string): Promise<void> {
  const db = await getChatDb();
  await db.delete('messages', messageId);
} 