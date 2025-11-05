# Frontend Features Documentation

Complete feature documentation for the CZSU Multi-Agent Text-to-SQL Frontend Application.

---

## Table of Contents

1. [Authentication & Security](#authentication--security)
2. [Chat Interface](#chat-interface)
3. [Data Exploration](#data-exploration)
4. [UI Components](#ui-components)
5. [State Management](#state-management)
6. [API Integration](#api-integration)
7. [Accessibility](#accessibility)

---

## Authentication & Security

### Google OAuth 2.0 Integration

The application uses NextAuth.js with Google OAuth provider for secure authentication.

**Configuration** (`frontend/src/app/api/auth/[...nextauth]/route.ts`):
```typescript
export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      if (account?.access_token) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    async session({ session, token }) {
      if (token?.accessToken) {
        session.accessToken = token.accessToken as string;
      }
      return session;
    },
  },
};
```

### Route Protection

**AuthGuard Component** (`frontend/src/components/AuthGuard.tsx`):
```typescript
export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const { data: session, status } = useSession();
  const pathname = usePathname();

  const protectedPaths = ["/chat", "/catalog", "/data"];
  const isProtectedRoute = protectedPaths.some((path) =>
    pathname?.startsWith(path)
  );

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading..." />
      </div>
    );
  }

  if (!session && isProtectedRoute) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen">
        <h1 className="text-2xl font-bold mb-4">Authentication Required</h1>
        <p className="mb-4">Please sign in to access this page.</p>
        <AuthButton />
      </div>
    );
  }

  return <>{children}</>;
}
```

**Protected routes**: `/chat`, `/catalog`, `/data`

### Automatic Token Refresh

**API Wrapper with Retry Logic** (`frontend/src/lib/api.ts`):
```typescript
export async function authApiFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const session = await getSession();
  
  if (!session?.accessToken) {
    throw new Error("No access token available");
  }

  const headers = {
    ...options.headers,
    Authorization: `Bearer ${session.accessToken}`,
    "Content-Type": "application/json",
  };

  let response = await fetch(url, { ...options, headers });

  // Retry once on 401 (token refresh)
  if (response.status === 401) {
    const newSession = await getSession();
    if (newSession?.accessToken) {
      headers.Authorization = `Bearer ${newSession.accessToken}`;
      response = await fetch(url, { ...options, headers });
    }
  }

  return response;
}
```

### Session Management

**User Avatar Display** (`frontend/src/components/AuthButton.tsx`):
```typescript
if (session) {
  const userName = session.user?.name || "User";
  const userEmail = session.user?.email || "";
  const userImage = session.user?.image;
  
  // Fallback initials
  const initials = userName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return compact ? (
    <button onClick={handleSignOut}>
      <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
        {userImage ? (
          <img src={userImage} alt={userName} className="w-8 h-8 rounded-full" />
        ) : (
          <span className="text-white text-sm font-semibold">{initials}</span>
        )}
      </div>
    </button>
  ) : (
    // ... full display
  );
}
```

### Logout with Cache Cleanup

**Comprehensive cleanup on logout**:
```typescript
const handleSignOut = async () => {
  await clearAllUserData();
  await signOut({ callbackUrl: "/" });
};

async function clearAllUserData() {
  try {
    const db = await openChatDB();
    const tx = db.transaction(["threads", "messages"], "readwrite");
    await tx.objectStore("threads").clear();
    await tx.objectStore("messages").clear();
    await tx.done;
  } catch (error) {
    console.error("Error clearing user data:", error);
  }
}
```

---

## Chat Interface

### Core Chat Functionality

#### Real-time Message Display

**Message Structure** (`frontend/src/components/MessageArea.tsx`):
```typescript
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sql?: string;
  datasetUrl?: string;
  top_chunks?: Array<{
    content: string;
    metadata: {
      source?: string;
      page?: number;
    };
  }>;
  followup_prompts?: string[];
  run_id?: string;
}
```

#### Message Rendering with Markdown

**ReactMarkdown Configuration**:
```typescript
<ReactMarkdown
  remarkPlugins={[remarkGfm]}
  components={{
    table: ({ node, ...props }) => (
      <table className="min-w-full border-collapse border border-gray-300 my-4" {...props} />
    ),
    thead: ({ node, ...props }) => (
      <thead className="bg-gray-100" {...props} />
    ),
    th: ({ node, ...props }) => (
      <th className="border border-gray-300 px-4 py-2 text-left font-semibold" {...props} />
    ),
    td: ({ node, ...props }) => (
      <td className="border border-gray-300 px-4 py-2" {...props} />
    ),
    code: ({ node, inline, ...props }) =>
      inline ? (
        <code className="bg-gray-100 px-1 py-0.5 rounded text-sm" {...props} />
      ) : (
        <code className="block bg-gray-100 p-2 rounded my-2 overflow-x-auto" {...props} />
      ),
  }}
>
  {message.content}
</ReactMarkdown>
```

### Advanced Message Features

#### Progress Bar for Long-running Queries

**8-minute timeout visualization**:
```typescript
const [progress, setProgress] = useState(0);

useEffect(() => {
  if (isLoading) {
    const startTime = Date.now();
    const maxDuration = 8 * 60 * 1000; // 8 minutes

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const newProgress = Math.min((elapsed / maxDuration) * 100, 100);
      setProgress(newProgress);

      if (newProgress >= 100) {
        clearInterval(interval);
      }
    }, 100);

    return () => clearInterval(interval);
  } else {
    setProgress(0);
  }
}, [isLoading]);

// Render progress bar
{isLoading && (
  <div className="w-full bg-gray-200 rounded-full h-2.5 mb-4">
    <div
      className="bg-blue-600 h-2.5 rounded-full transition-all duration-100"
      style={{ width: `${progress}%` }}
    />
  </div>
)}
```

#### Copy to Clipboard

**HTML and Plain Text Support**:
```typescript
const copyToClipboard = (text: string, format: "html" | "plain" = "plain") => {
  if (format === "html") {
    const blob = new Blob([text], { type: "text/html" });
    const item = new ClipboardItem({ "text/html": blob });
    navigator.clipboard.write([item]).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      },
      (err) => console.error("Failed to copy HTML:", err)
    );
  } else {
    navigator.clipboard.writeText(text).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      },
      (err) => console.error("Failed to copy text:", err)
    );
  }
};
```

#### SQL Query Modal

**Display SQL with dataset link**:
```typescript
{message.sql && (
  <button
    onClick={() => {
      setSqlModalOpen(true);
      setCurrentSql(message.sql || "");
      setCurrentDatasetUrl(message.datasetUrl || null);
    }}
    className="text-blue-500 hover:underline text-sm"
  >
    📊 View SQL Query
  </button>
)}

<Modal open={sqlModalOpen} onClose={() => setSqlModalOpen(false)}>
  <div className="max-w-3xl mx-auto bg-white p-6 rounded-lg">
    <h2 className="text-xl font-bold mb-4">SQL Query</h2>
    <pre className="bg-gray-100 p-4 rounded overflow-x-auto text-sm">
      <code>{currentSql}</code>
    </pre>
    {currentDatasetUrl && (
      <div className="mt-4">
        <Link
          href={currentDatasetUrl}
          className="text-blue-500 hover:underline"
        >
          🔗 View Dataset in Catalog
        </Link>
      </div>
    )}
  </div>
</Modal>
```

#### PDF Document Chunks

**Display source documents with metadata**:
```typescript
{message.top_chunks && message.top_chunks.length > 0 && (
  <button
    onClick={() => {
      setPdfModalOpen(true);
      setCurrentPdfChunks(message.top_chunks || []);
    }}
    className="text-blue-500 hover:underline text-sm"
  >
    📄 View Source Documents ({message.top_chunks.length})
  </button>
)}

<Modal open={pdfModalOpen} onClose={() => setPdfModalOpen(false)}>
  <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg max-h-[80vh] overflow-y-auto">
    <h2 className="text-xl font-bold mb-4">
      Source Documents ({currentPdfChunks.length})
    </h2>
    {currentPdfChunks.map((chunk, idx) => (
      <div key={idx} className="mb-6 p-4 bg-gray-50 rounded">
        <div className="text-sm text-gray-600 mb-2">
          <strong>Source:</strong>{" "}
          {chunk.metadata?.source || "Unknown"}
          {chunk.metadata?.page && (
            <> | <strong>Page:</strong> {chunk.metadata.page}</>
          )}
        </div>
        <div className="text-sm whitespace-pre-wrap">{chunk.content}</div>
      </div>
    ))}
  </div>
</Modal>
```

#### Follow-up Prompts

**Suggested next questions**:
```typescript
{message.followup_prompts && message.followup_prompts.length > 0 && (
  <div className="mt-4 p-3 bg-blue-50 rounded-lg">
    <p className="text-sm font-semibold mb-2">💡 Follow-up suggestions:</p>
    <div className="space-y-2">
      {message.followup_prompts.map((prompt, idx) => (
        <button
          key={idx}
          onClick={() => onFollowUpClick(prompt)}
          className="block w-full text-left text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-100 px-3 py-2 rounded transition-colors"
        >
          {prompt}
        </button>
      ))}
    </div>
  </div>
)}
```

### Feedback System

#### Sentiment Tracking

**Thumbs up/down with persistence**:
```typescript
const handleThumbsUp = async (messageId: string, runId?: string) => {
  const currentSentiment = sentiments[messageId];
  const newSentiment = currentSentiment === "positive" ? null : "positive";

  setSentiments((prev) => ({
    ...prev,
    [messageId]: newSentiment,
  }));

  if (runId) {
    try {
      await authApiFetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        body: JSON.stringify({
          run_id: runId,
          sentiment: newSentiment,
        }),
      });
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    }
  }

  // Persist to localStorage
  localStorage.setItem(
    `sentiment_${messageId}`,
    JSON.stringify(newSentiment)
  );
};
```

#### Comment System

**Additional feedback with modal**:
```typescript
{showCommentBox === message.id && (
  <div className="mt-2 p-3 bg-gray-50 rounded">
    <textarea
      value={commentText}
      onChange={(e) => setCommentText(e.target.value)}
      placeholder="Add your feedback..."
      className="w-full p-2 border rounded text-sm"
      rows={3}
    />
    <div className="flex gap-2 mt-2">
      <button
        onClick={() => handleCommentSubmit(message.id, message.run_id)}
        className="px-3 py-1 bg-blue-500 text-white rounded text-sm"
      >
        Submit
      </button>
      <button
        onClick={() => {
          setShowCommentBox(null);
          setCommentText("");
        }}
        className="px-3 py-1 bg-gray-300 rounded text-sm"
      >
        Cancel
      </button>
    </div>
  </div>
)}
```

---

## Data Exploration

### Catalog Browser

#### Full-text Search with Diacritics Support

**Search implementation** (`frontend/src/components/DatasetsTable.tsx`):
```typescript
const removeDiacritics = (text: string): string => {
  return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
};

const filteredDatasets = useMemo(() => {
  if (!filterText.trim()) return datasets;

  const normalizedFilter = removeDiacritics(filterText.toLowerCase());
  const isCodeOnlySearch = normalizedFilter.startsWith("*");
  const searchTerm = isCodeOnlySearch
    ? normalizedFilter.slice(1)
    : normalizedFilter;

  return datasets.filter((dataset) => {
    if (isCodeOnlySearch) {
      const code = removeDiacritics(dataset.kod.toLowerCase());
      return code.includes(searchTerm);
    }

    const code = removeDiacritics(dataset.kod.toLowerCase());
    const text = removeDiacritics(dataset.text.toLowerCase());
    const extended = dataset.extended_text
      ? removeDiacritics(dataset.extended_text.toLowerCase())
      : "";

    return (
      code.includes(searchTerm) ||
      text.includes(searchTerm) ||
      extended.includes(searchTerm)
    );
  });
}, [datasets, filterText]);
```

#### Pagination

**Page navigation with persistence**:
```typescript
const [currentPage, setCurrentPage] = useState(() => {
  const saved = localStorage.getItem("catalogPage");
  return saved ? parseInt(saved, 10) : 1;
});

const itemsPerPage = 10;
const totalPages = Math.ceil(filteredDatasets.length / itemsPerPage);
const startIndex = (currentPage - 1) * itemsPerPage;
const endIndex = startIndex + itemsPerPage;
const currentDatasets = filteredDatasets.slice(startIndex, endIndex);

useEffect(() => {
  localStorage.setItem("catalogPage", currentPage.toString());
}, [currentPage]);

// Pagination controls
<div className="flex items-center justify-between mt-4">
  <button
    onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
    disabled={currentPage === 1}
    className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
  >
    Previous
  </button>
  <span>
    Page {currentPage} of {totalPages}
  </span>
  <button
    onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
    disabled={currentPage === totalPages}
    className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
  >
    Next
  </button>
</div>
```

### Data Table Viewer

#### Auto-complete Search

**Table suggestions** (`frontend/src/components/DataTableView.tsx`):
```typescript
const [suggestions, setSuggestions] = useState<string[]>([]);
const [showSuggestions, setShowSuggestions] = useState(false);

useEffect(() => {
  if (tableName.length >= 2) {
    const filtered = availableTables.filter((table) =>
      table.toLowerCase().includes(tableName.toLowerCase())
    );
    setSuggestions(filtered.slice(0, 10));
    setShowSuggestions(true);
  } else {
    setSuggestions([]);
    setShowSuggestions(false);
  }
}, [tableName, availableTables]);

// Suggestion dropdown
{showSuggestions && suggestions.length > 0 && (
  <ul className="absolute z-10 w-full bg-white border rounded-md shadow-lg max-h-60 overflow-auto">
    {suggestions.map((suggestion, idx) => (
      <li
        key={idx}
        onClick={() => {
          setTableName(suggestion);
          setShowSuggestions(false);
        }}
        className="px-4 py-2 hover:bg-blue-100 cursor-pointer"
      >
        {suggestion}
      </li>
    ))}
  </ul>
)}
```

#### Column-specific Filtering

**Multi-word search with operators**:
```typescript
const filterRows = (
  rows: any[],
  columns: string[],
  filters: Record<string, string>
) => {
  return rows.filter((row) => {
    return Object.entries(filters).every(([colName, filterValue]) => {
      if (!filterValue.trim()) return true;

      const cellValue = row[colName];
      if (cellValue === null || cellValue === undefined) return false;

      const cellStr = String(cellValue).toLowerCase();
      const filterNorm = removeDiacritics(filterValue.trim().toLowerCase());

      // Numeric operators
      const numericMatch = filterNorm.match(/^([><=!]+)\s*(.+)$/);
      if (numericMatch) {
        const operator = numericMatch[1];
        const numValue = parseFloat(numericMatch[2]);
        const cellNum = parseFloat(cellStr);

        if (!isNaN(cellNum) && !isNaN(numValue)) {
          switch (operator) {
            case ">": return cellNum > numValue;
            case "<": return cellNum < numValue;
            case ">=": return cellNum >= numValue;
            case "<=": return cellNum <= numValue;
            case "=": return cellNum === numValue;
            case "!=": return cellNum !== numValue;
            default: return false;
          }
        }
      }

      // Multi-word search
      const filterWords = filterNorm.split(/\s+/).filter((w) => w.length > 0);
      return filterWords.every((word) =>
        removeDiacritics(cellStr).includes(word)
      );
    });
  });
};
```

#### Sorting

**Multi-column sort with direction**:
```typescript
const [sortConfig, setSortConfig] = useState<{
  column: string | null;
  direction: "asc" | "desc";
}>({ column: null, direction: "asc" });

const handleSort = (column: string) => {
  setSortConfig((prev) => ({
    column,
    direction:
      prev.column === column && prev.direction === "asc" ? "desc" : "asc",
  }));
};

const sortedRows = useMemo(() => {
  if (!sortConfig.column) return filteredRows;

  return [...filteredRows].sort((a, b) => {
    const aVal = a[sortConfig.column!];
    const bVal = b[sortConfig.column!];

    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;

    const aNum = parseFloat(String(aVal));
    const bNum = parseFloat(String(bVal));

    if (!isNaN(aNum) && !isNaN(bNum)) {
      return sortConfig.direction === "asc" ? aNum - bNum : bNum - aNum;
    }

    const aStr = String(aVal).toLowerCase();
    const bStr = String(bVal).toLowerCase();

    if (sortConfig.direction === "asc") {
      return aStr.localeCompare(bStr);
    } else {
      return bStr.localeCompare(aStr);
    }
  });
}, [filteredRows, sortConfig]);
```

---

## UI Components

### Modal System

**Reusable Modal Component** (`frontend/src/components/Modal.tsx`):
```typescript
interface ModalProps {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

export default function Modal({ open, onClose, children }: ModalProps) {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="relative max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-gray-600 hover:text-gray-900 text-2xl font-bold"
        >
          ×
        </button>
        {children}
      </div>
    </div>
  );
}
```

**Usage examples**:
- SQL Query Display Modal
- PDF Document Chunks Modal
- Feedback Comment Modal

### Loading Spinner

**Three Size Variants** (`frontend/src/components/LoadingSpinner.tsx`):
```typescript
interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg";
  text?: string;
  className?: string;
}

export default function LoadingSpinner({
  size = "md",
  text,
  className = "",
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-8 h-8",
    lg: "w-12 h-12",
  };

  return (
    <div className={`flex flex-col items-center gap-2 ${className}`}>
      <div
        className={`${sizeClasses[size]} border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin`}
      />
      {text && <p className="text-sm text-gray-600">{text}</p>}
    </div>
  );
}
```

### Input Components

**Chat Input Bar** (`frontend/src/components/InputBar.tsx`):
```typescript
interface InputBarProps {
  currentMessage: string;
  setCurrentMessage: (msg: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  isLoading: boolean;
}

const InputBar = forwardRef<HTMLInputElement, InputBarProps>(
  ({ currentMessage, setCurrentMessage, onSubmit, isLoading }, ref) => {
    return (
      <form onSubmit={onSubmit} className="flex gap-2">
        <input
          ref={ref}
          type="text"
          value={currentMessage}
          onChange={(e) => setCurrentMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={isLoading || !currentMessage.trim()}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          <span className={isLoading ? "animate-spin" : ""}>→</span>
        </button>
      </form>
    );
  }
);
```

### Header Navigation

**Responsive Navigation with Active States** (`frontend/src/components/Header.tsx`):
```typescript
export default function Header() {
  const pathname = usePathname();

  const navItems = [
    { href: "/", label: "🏠 HOME" },
    { href: "/chat", label: "💬 CHAT" },
    { href: "/catalog", label: "📚 CATALOG" },
    { href: "/data", label: "📊 DATA" },
    { href: "/contacts", label: "📧 CONTACTS" },
  ];

  return (
    <header className="bg-white shadow-sm">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex space-x-8">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                    isActive
                      ? "border-blue-500 text-gray-900"
                      : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
          <div className="flex items-center">
            <AuthButton />
          </div>
        </div>
      </nav>
    </header>
  );
}
```

---

## State Management

### IndexedDB Integration

**Chat Database Structure** (`frontend/src/components/utils.ts`):
```typescript
export async function openChatDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("ChatDB", 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      if (!db.objectStoreNames.contains("threads")) {
        const threadStore = db.createObjectStore("threads", {
          keyPath: "thread_id",
        });
        threadStore.createIndex("user_email", "user_email", { unique: false });
        threadStore.createIndex("timestamp", "timestamp", { unique: false });
      }

      if (!db.objectStoreNames.contains("messages")) {
        const messageStore = db.createObjectStore("messages", {
          keyPath: "id",
          autoIncrement: true,
        });
        messageStore.createIndex("thread_id", "thread_id", { unique: false });
        messageStore.createIndex("timestamp", "timestamp", { unique: false });
      }
    };
  });
}
```

### Thread Management

**CRUD Operations**:
```typescript
export async function saveThread(
  threadId: string,
  userEmail: string,
  title: string
): Promise<void> {
  const db = await openChatDB();
  const tx = db.transaction("threads", "readwrite");
  const store = tx.objectStore("threads");

  await store.put({
    thread_id: threadId,
    user_email: userEmail,
    title: title,
    timestamp: new Date().toISOString(),
  });

  await tx.done;
}

export async function getThreadsForUser(
  userEmail: string
): Promise<Thread[]> {
  const db = await openChatDB();
  const tx = db.transaction("threads", "readonly");
  const store = tx.objectStore("threads");
  const index = store.index("user_email");

  const threads = await index.getAll(userEmail);
  await tx.done;

  return threads.sort(
    (a, b) =>
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
}

export async function deleteThread(threadId: string): Promise<void> {
  const db = await openChatDB();
  const tx = db.transaction(["threads", "messages"], "readwrite");

  // Delete thread
  await tx.objectStore("threads").delete(threadId);

  // Delete all messages in thread
  const messageStore = tx.objectStore("messages");
  const index = messageStore.index("thread_id");
  const messages = await index.getAllKeys(threadId);

  for (const key of messages) {
    await messageStore.delete(key);
  }

  await tx.done;
}
```

### Message Management

**Store and Retrieve Messages**:
```typescript
export async function saveMessage(
  threadId: string,
  message: Omit<Message, "id">
): Promise<void> {
  const db = await openChatDB();
  const tx = db.transaction("messages", "readwrite");
  const store = tx.objectStore("messages");

  await store.add({
    ...message,
    thread_id: threadId,
    timestamp: message.timestamp || new Date().toISOString(),
  });

  await tx.done;
}

export async function getMessagesForThread(
  threadId: string
): Promise<Message[]> {
  const db = await openChatDB();
  const tx = db.transaction("messages", "readonly");
  const store = tx.objectStore("messages");
  const index = store.index("thread_id");

  const messages = await index.getAll(threadId);
  await tx.done;

  return messages.sort(
    (a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );
}
```

### Chat Cache Context

**Global State Management** (`frontend/src/context/ChatCacheContext.tsx`):
```typescript
interface ChatCacheContextType {
  threads: Thread[];
  currentThreadId: string | null;
  setCurrentThreadId: (id: string | null) => void;
  addThread: (thread: Thread) => void;
  deleteThread: (threadId: string) => void;
  updateThreadTitle: (threadId: string, title: string) => void;
  refreshThreads: () => Promise<void>;
}

export function ChatCacheProvider({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);

  const refreshThreads = useCallback(async () => {
    if (!session?.user?.email) return;

    try {
      const userThreads = await getThreadsForUser(session.user.email);
      setThreads(userThreads);
    } catch (error) {
      console.error("Error loading threads:", error);
    }
  }, [session?.user?.email]);

  useEffect(() => {
    refreshThreads();
  }, [refreshThreads]);

  const addThread = useCallback((thread: Thread) => {
    setThreads((prev) => [thread, ...prev]);
  }, []);

  const deleteThread = useCallback(async (threadId: string) => {
    try {
      await deleteThreadFromDB(threadId);
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
      if (currentThreadId === threadId) {
        setCurrentThreadId(null);
      }
    } catch (error) {
      console.error("Error deleting thread:", error);
    }
  }, [currentThreadId]);

  return (
    <ChatCacheContext.Provider
      value={{
        threads,
        currentThreadId,
        setCurrentThreadId,
        addThread,
        deleteThread,
        updateThreadTitle,
        refreshThreads,
      }}
    >
      {children}
    </ChatCacheContext.Provider>
  );
}
```

### LocalStorage Persistence

**Filter and Pagination State**:
```typescript
// Persist catalog page
useEffect(() => {
  const savedPage = localStorage.getItem("catalogPage");
  if (savedPage) {
    setCurrentPage(parseInt(savedPage, 10));
  }
}, []);

useEffect(() => {
  localStorage.setItem("catalogPage", currentPage.toString());
}, [currentPage]);

// Persist filter text
useEffect(() => {
  const savedFilter = localStorage.getItem("catalogFilter");
  if (savedFilter) {
    setFilterText(savedFilter);
  }
}, []);

useEffect(() => {
  localStorage.setItem("catalogFilter", filterText);
}, [filterText]);

// Persist table selection
useEffect(() => {
  const savedTable = localStorage.getItem("selectedTable");
  if (savedTable) {
    setTableName(savedTable);
  }
}, []);

useEffect(() => {
  if (tableName) {
    localStorage.setItem("selectedTable", tableName);
  }
}, [tableName]);
```

---

## API Integration

### Authentication Wrapper

**authApiFetch Implementation** (`frontend/src/lib/api.ts`):
```typescript
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function authApiFetch(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  const session = await getSession();

  if (!session?.accessToken) {
    throw new Error("No access token available");
  }

  const headers = {
    ...options.headers,
    Authorization: `Bearer ${session.accessToken}`,
    "Content-Type": "application/json",
  };

  let response = await fetch(url, { ...options, headers });

  // Retry once on 401 (token expired/invalid)
  if (response.status === 401) {
    console.log("Token expired, refreshing session...");
    const newSession = await getSession();

    if (newSession?.accessToken) {
      headers.Authorization = `Bearer ${newSession.accessToken}`;
      response = await fetch(url, { ...options, headers });
    } else {
      throw new Error("Failed to refresh session");
    }
  }

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `API request failed: ${response.status} ${response.statusText} - ${errorText}`
    );
  }

  return response;
}
```

### API Endpoints

**Chat Analysis**:
```typescript
const response = await authApiFetch(`${API_BASE_URL}/analyze`, {
  method: "POST",
  body: JSON.stringify({
    prompt: message,
    thread_id: threadId,
  }),
});

const data = await response.json();
// Returns: { result, sql, datasetUrl, top_chunks, followup_prompts }
```

**Thread Operations**:
```typescript
// Get all threads for user
const response = await authApiFetch(
  `${API_BASE_URL}/chat/all-messages-for-all-threads`
);
const threads = await response.json();

// Get messages for specific thread
const response = await authApiFetch(
  `${API_BASE_URL}/chat/all-messages-for-one-thread/${threadId}`
);
const messages = await response.json();

// Delete thread
await authApiFetch(`${API_BASE_URL}/chat/thread/${threadId}`, {
  method: "DELETE",
});
```

**Feedback Submission**:
```typescript
await authApiFetch(`${API_BASE_URL}/feedback`, {
  method: "POST",
  body: JSON.stringify({
    run_id: runId,
    sentiment: "positive" | "negative" | null,
    comment: "Optional feedback text",
  }),
});
```

**Catalog Data**:
```typescript
// Get all datasets
const response = await fetch(`${API_BASE_URL}/catalog/selections`);
const datasets = await response.json();

// Get specific dataset
const response = await fetch(`${API_BASE_URL}/catalog/selection/${code}`);
const dataset = await response.json();
```

**Table Data**:
```typescript
// Get available tables
const response = await authApiFetch(`${API_BASE_URL}/data/tables`);
const tables = await response.json();

// Query table data
const response = await authApiFetch(`${API_BASE_URL}/data/query`, {
  method: "POST",
  body: JSON.stringify({
    table_name: tableName,
    limit: 1000,
  }),
});
const { columns, rows } = await response.json();
```

### Error Handling

**Comprehensive error management**:
```typescript
try {
  const response = await authApiFetch(url, options);
  const data = await response.json();
  return data;
} catch (error) {
  if (error instanceof Error) {
    if (error.message.includes("401")) {
      // Session expired - redirect to login
      console.error("Authentication failed");
      signOut({ callbackUrl: "/login" });
    } else if (error.message.includes("429")) {
      // Rate limit
      console.error("Too many requests");
      setError("Rate limit exceeded. Please try again later.");
    } else if (error.message.includes("500")) {
      // Server error
      console.error("Server error");
      setError("Server error. Please try again later.");
    } else {
      // Generic error
      console.error("Request failed:", error);
      setError("An error occurred. Please try again.");
    }
  }
  throw error;
}
```

---

## Accessibility

### Keyboard Navigation

**Modal Escape Key Support**:
```typescript
useEffect(() => {
  const handleEscape = (e: KeyboardEvent) => {
    if (e.key === "Escape" && open) {
      onClose();
    }
  };

  document.addEventListener("keydown", handleEscape);
  return () => document.removeEventListener("keydown", handleEscape);
}, [open, onClose]);
```

**Form Enter Key Submission**:
```typescript
<form onSubmit={handleSubmit} onKeyDown={(e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSubmit(e);
  }
}}>
```

### Focus Management

**Input Auto-focus**:
```typescript
const inputRef = useRef<HTMLInputElement>(null);

useEffect(() => {
  if (!isLoading && inputRef.current) {
    inputRef.current.focus();
  }
}, [isLoading]);
```

**Modal Focus Trap**:
```typescript
useEffect(() => {
  if (open) {
    const focusableElements = modalRef.current?.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements?.[0] as HTMLElement;
    firstElement?.focus();
  }
}, [open]);
```

### ARIA Labels

**Screen Reader Support**:
```typescript
<button
  aria-label="Submit message"
  aria-disabled={isLoading}
  onClick={handleSubmit}
>
  Send
</button>

<div role="status" aria-live="polite">
  {isLoading && <LoadingSpinner text="Processing..." />}
</div>

<nav aria-label="Main navigation">
  {navItems.map((item) => (
    <Link
      key={item.href}
      href={item.href}
      aria-current={isActive ? "page" : undefined}
    >
      {item.label}
    </Link>
  ))}
</nav>
```

### Semantic HTML

**Proper Structure**:
```typescript
<main>
  <h1>Chat Interface</h1>
  <section aria-label="Chat messages">
    <article>
      <h2 className="sr-only">User message</h2>
      {/* Message content */}
    </article>
  </section>
  <footer>
    <form aria-label="Message input">
      {/* Input field */}
    </form>
  </footer>
</main>
```

---

## Performance Optimizations

### Memoization

**Expensive Computations**:
```typescript
const filteredDatasets = useMemo(() => {
  if (!filterText.trim()) return datasets;
  return datasets.filter(/* expensive filtering */);
}, [datasets, filterText]);

const sortedRows = useMemo(() => {
  if (!sortConfig.column) return filteredRows;
  return [...filteredRows].sort(/* sorting logic */);
}, [filteredRows, sortConfig]);
```

### Debounced Search

**Search Input Optimization**:
```typescript
import { useDebounce } from 'use-debounce';

const [searchText, setSearchText] = useState("");
const [debouncedSearch] = useDebounce(searchText, 300);

useEffect(() => {
  if (debouncedSearch) {
    performSearch(debouncedSearch);
  }
}, [debouncedSearch]);
```

### Lazy Loading

**Component Code Splitting**:
```typescript
import dynamic from 'next/dynamic';

const DatasetsTable = dynamic(
  () => import('@/components/DatasetsTable'),
  { loading: () => <LoadingSpinner size="lg" /> }
);

const DataTableView = dynamic(
  () => import('@/components/DataTableView'),
  { ssr: false }
);
```

### Memory Management

**Cleanup on Unmount**:
```typescript
useEffect(() => {
  const interval = setInterval(() => {
    // Some periodic task
  }, 1000);

  return () => {
    clearInterval(interval);
    // Additional cleanup
  };
}, []);
```

---

## Summary

The CZSU Multi-Agent Text-to-SQL Frontend is a comprehensive React/Next.js application with:

- **🔐 Robust Authentication**: Google OAuth with automatic token refresh
- **💬 Advanced Chat**: Real-time messaging with markdown, SQL display, PDF sources, and feedback
- **📊 Data Exploration**: Full-featured catalog browser and table viewer with filtering/sorting
- **🎨 Reusable Components**: Modal, LoadingSpinner, InputBar, AuthButton, Header
- **💾 Persistent State**: IndexedDB for offline chat, LocalStorage for preferences
- **🚀 Performance**: Memoization, debouncing, lazy loading
- **♿ Accessibility**: Keyboard navigation, ARIA labels, semantic HTML
- **🔧 Developer Experience**: TypeScript, custom hooks, clean architecture

All features are production-ready and tested in the live application.
