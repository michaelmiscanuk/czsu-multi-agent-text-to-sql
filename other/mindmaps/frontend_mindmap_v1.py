import os

from graphviz import Digraph

# Define the mindmap structure as a nested dictionary
mindmap = {
    "Next.js Application": {
        "Configuration": {
            "Package management (package.json)": {
                "Dependencies": [
                    "next@15.3.0 - React framework with SSR/SSG",
                    "react@19.0.0 & react-dom@19.0.0 - UI library",
                    "next-auth@4.24.11 - Authentication with Google OAuth",
                    "idb@7.1.1 - IndexedDB wrapper for client storage",
                    "markdown-to-jsx@7.7.12 - Markdown rendering",
                    "uuid@11.1.0 - Unique ID generation",
                    "swagger-ui-react@5.30.2 - API documentation",
                ],
                "Dev Dependencies": [
                    "tailwindcss@4 - Utility-first CSS framework",
                    "@tailwindcss/postcss@4 - PostCSS integration",
                    "typescript@5 - Type safety",
                    "@types/node, @types/react, @types/react-dom - Type definitions",
                ],
                "Summary": "Modern Next.js stack with React 19, NextAuth for authentication, TailwindCSS 4 for styling, and TypeScript for type safety.",
            },
            "TypeScript config (tsconfig.json)": {
                "Details": [
                    "target: ES2017 for modern JavaScript",
                    "strict mode enabled for type checking",
                    "Path aliases: @/* maps to ./src/*",
                    "Module resolution: bundler mode for Next.js",
                    "Incremental compilation enabled",
                ],
                "Summary": "Strict TypeScript configuration with path aliases and Next.js plugin for enhanced DX.",
            },
            "Next.js config (next.config.ts)": {
                "Details": [
                    "Minimal configuration - using defaults",
                    "No custom webpack/babel overrides",
                ],
                "Summary": "Clean Next.js 15 configuration relying on framework defaults.",
            },
            "Deployment config (vercel.json)": {
                "Details": [
                    "17 API rewrites to Railway backend: /api/* → backend URL",
                    "Routes: /analyze, /chat-threads, /chat/:path*, /feedback, /sentiment",
                    "Data routes: /catalog, /data-tables, /data-table",
                    "Debug routes: /debug/:path*, /agents/:path*",
                    "Excludes NextAuth routes with ((?!auth).*) pattern",
                ],
                "Summary": "Vercel deployment with comprehensive API proxying to Railway backend, enabling seamless frontend-backend communication.",
            },
            "Summary": "Next.js 15 + React 19 + TypeScript stack with TailwindCSS 4, NextAuth, and production deployment via Vercel proxying to Railway backend.",
        },
        "Routing & Layout": {
            "App Router structure": {
                "Root layout (app/layout.tsx)": {
                    "Details": [
                        "Metadata: title and description for SEO",
                        "HTML lang='en' with antialiased body",
                        "SessionProviderWrapper for NextAuth context",
                        "ClientLayout for client-side navigation logic",
                        "suppressHydrationWarning to avoid SSR/client mismatch",
                    ],
                    "Summary": "Server component providing metadata and wrapping app in authentication + client layout.",
                },
                "Client layout (app/ClientLayout.tsx)": {
                    "Details": [
                        "PUBLIC_ROUTES: /, /contacts, /login, /terms-of-use",
                        "PROTECTED_ROUTES: /chat, /catalog, /data",
                        "useSession + usePathname for auth state + route detection",
                        "Auto-redirect unauthenticated users to /login",
                        "ChatCacheProvider wrapping for global state",
                        "Conditional footer hiding for chat/catalog/data pages",
                        "Sticky Header with z-50 for always-visible navigation",
                    ],
                    "Summary": "Client-side layout managing authentication redirects, global cache provider, header/footer visibility, and gradient background.",
                },
                "Summary": "Dual-layout approach: server layout for metadata + wrapping, client layout for auth logic + UI structure.",
            },
            "Pages": {
                "Home (/)": {
                    "Details": [
                        "Welcome message with CZSU info",
                        "Links to CZSU API and PDF documentation",
                        "Centered card layout with blue gradient background",
                    ],
                    "Summary": "Landing page introducing the CZSU Data Explorer with external resource links.",
                },
                "Chat (/chat)": {
                    "Core functionality": {
                        "Details": [
                            "v4 uuidv4() for thread_id and run_id generation",
                            "ChatCacheProvider hooks for threads, messages, pagination",
                            "useSession for user authentication and email",
                            "useState for local UI state (currentMessage, modals, loading)",
                            "useInfiniteScroll for thread pagination with observer",
                        ],
                        "Summary": "Primary chat interface with thread management, message display, and infinite scroll pagination.",
                    },
                    "Thread management": {
                        "Details": [
                            "NEW CHAT: Creates thread with UUID, updates cache, fetches initial prompts",
                            "RENAME: Updates thread title in cache via updateThread()",
                            "DELETE: Calls /chat/{thread_id} DELETE, removes from cache, clears activeThreadId",
                            "PAGINATION: loadInitialThreads() + loadMoreThreads() with page/limit",
                            "Infinite scroll observer triggers loadMoreThreads()",
                        ],
                        "Summary": "Full thread lifecycle with create, rename, delete, and paginated loading via API + cache.",
                    },
                    "Message flow": {
                        "Send sequence": [
                            "1. Validate message + user email + no concurrent loading (checkUserLoadingState)",
                            "2. Clear input + draft localStorage",
                            "3. Set local + context loading states",
                            "4. Set cross-tab loading state (setUserLoadingState)",
                            "5. Create/reuse thread, update title if needed",
                            "6. Add user message to cache with isLoading=true",
                            "7. Generate run_id, call /analyze with prompt + thread_id + run_id",
                            "8. On success: update message with final_answer, datasets_used, followup_prompts",
                            "9. CRITICAL FIX: Sync with backend via /chat/all-messages-for-one-thread/{threadId}",
                            "10. On error: Recovery mechanism checks PostgreSQL for saved response",
                            "11. Finally: clear all loading states + run_id",
                        ],
                        "Recovery mechanism": {
                            "Details": [
                                "checkForNewMessagesAfterTimeout() waits 1s then queries backend",
                                "Compares message count before/after for new content",
                                "Replaces frontend cache with authoritative backend data if found",
                                "Falls back to error message if recovery fails",
                            ],
                            "Summary": "Resilient message handling with backend sync to prevent data loss during memory pressure.",
                        },
                        "Summary": "Robust message sending with optimistic UI, backend sync, cross-tab loading prevention, and error recovery.",
                    },
                    "Cancellation": {
                        "Details": [
                            "handleStopExecution() sends /stop-execution with thread_id + run_id",
                            "Optimistic UI: immediately clears loading states + updates loading message",
                            "Background API call for actual cancellation (non-blocking)",
                        ],
                        "Summary": "Instant UI feedback for stop requests with asynchronous backend cancellation.",
                    },
                    "LocalStorage persistence": {
                        "Details": [
                            "czsu-draft-message: current input text",
                            "czsu-last-active-chat: last active thread ID",
                            "czsu-chat-cache: full cache managed by ChatCacheContext",
                        ],
                        "Summary": "Client-side persistence for draft messages, active thread, and full chat cache.",
                    },
                    "Summary": "Feature-rich chat page with thread CRUD, paginated loading, message send/receive, cancellation, localStorage persistence, and recovery mechanisms.",
                },
                "Catalog (/catalog)": {
                    "Details": [
                        "DatasetsTable component with pagination",
                        "onRowClick navigates to /data?table={selection_code}",
                        "unified-white-block-system styling with table-container",
                    ],
                    "Summary": "Browse CZSU datasets with pagination and click-through to data view.",
                },
                "Data (/data)": {
                    "State management": {
                        "Details": [
                            "search, selectedTable, columns, rows, selectedColumn, columnFilters",
                            "All state lifted to DataPage for localStorage sync",
                            "pendingTableSearch from URL param ?table=...",
                            "Separate keys: czsu-data-search, czsu-data-selectedTable, etc.",
                        ],
                        "Summary": "Complex state management with URL params, localStorage, and lifted state for table exploration.",
                    },
                    "Features": {
                        "Details": [
                            "Auto-complete search with diacritics removal",
                            "* prefix searches only in selection_code",
                            "Column-specific filtering with numeric operators (>, <, >=, <=, !=, =)",
                            "Sortable columns (asc/desc/none) with visual indicators",
                            "Click table code badge to navigate to catalog with prefilled filter",
                        ],
                        "Summary": "Advanced table view with multi-filter, sort, search, and cross-navigation to catalog.",
                    },
                    "Summary": "Data exploration page with smart search, filtering, sorting, and seamless navigation.",
                },
                "Login (/login)": {
                    "Details": [
                        "AuthButton component for Google OAuth sign-in",
                        "Auto-redirect to /chat if authenticated",
                        "Simple card layout with user icon",
                    ],
                    "Summary": "Dedicated login page with Google OAuth and auto-navigation for authenticated users.",
                },
                "Contacts (/contacts)": {
                    "Details": [
                        "Static contact info: name, email, LinkedIn, GitHub",
                        "About me section describing interests",
                        "Links to external profiles",
                    ],
                    "Summary": "Static contact information page.",
                },
                "Summary": "App Router with 6 main pages: Home, Chat, Catalog, Data, Login, Contacts - each with distinct functionality and state management.",
            },
            "Summary": "Next.js App Router architecture with server/client layout separation, protected routes, and feature-rich pages.",
        },
        "Components": {
            "Layout components": {
                "Header": {
                    "Details": [
                        "Menu items: HOME, CHAT, CATALOG, DATA, CONTACTS",
                        "usePathname for active route highlighting",
                        "Logo click navigates to /chat if authenticated, else /",
                        "AuthButton for login/logout",
                        "Sticky positioning with shadow and gradient border",
                    ],
                    "Summary": "Persistent header with navigation menu, logo, and auth controls.",
                },
                "AuthButton": {
                    "Details": [
                        "Compact mode for header, full mode for login page",
                        "signIn('google') for OAuth login",
                        "signOut() with clearCacheForUserChange() cleanup",
                        "User avatar from session.user.image or initials",
                        "Displays user name or email",
                    ],
                    "Summary": "Dual-mode auth button handling Google sign-in/out with cache cleanup and user display.",
                },
                "SessionProviderWrapper": {
                    "Details": [
                        "Client component wrapping SessionProvider from next-auth/react",
                        "Provides session context to entire app",
                    ],
                    "Summary": "NextAuth session provider wrapper for authentication context.",
                },
                "AuthGuard": {
                    "Details": [
                        "PUBLIC_ROUTES and PROTECTED_ROUTES logic",
                        "Shows loading state while session initializes",
                        "Renders children for public routes or authenticated users",
                        "Shows login prompt with AuthButton for protected routes",
                        "Fallback redirect message for other unauthenticated access",
                    ],
                    "Summary": "Route guard component protecting pages based on authentication status.",
                },
                "Summary": "Core layout components for navigation, authentication UI, and route protection.",
            },
            "Chat components": {
                "MessageArea": {
                    "Message rendering": {
                        "Details": [
                            "MarkdownText component with containsMarkdown() detection",
                            "Custom markdown overrides for p, ul, ol, li, tables, code, etc.",
                            "Copy button (rich text + plain text via ClipboardItem)",
                            "User prompts on right, AI responses on left",
                            "Rerun button for user prompts (when not loading)",
                            "SimpleProgressBar for loading messages with 8-minute estimate",
                        ],
                        "Summary": "Sophisticated message rendering with markdown support, copy functionality, and progress tracking.",
                    },
                    "Datasets & actions": {
                        "Details": [
                            "datasets_used badges link to /data?table={code}",
                            "SQL button opens modal with queries_and_results",
                            "PDF button opens modal with top_chunks",
                            "Follow-up prompts as clickable badges (only for latest message)",
                        ],
                        "Summary": "Rich message metadata display with navigation, modals, and follow-up suggestions.",
                    },
                    "Feedback system": {
                        "FeedbackComponent": {
                            "Details": [
                                "Thumbs up/down for sentiment (true/false)",
                                "Comment bubble with dropdown textarea",
                                "Sentiment submission via /sentiment endpoint",
                                "LangSmith feedback via /feedback endpoint",
                                "Persistent feedback stored in czsu-persistent-feedback localStorage",
                                "Shows 'selected: 👍/👎' after sentiment chosen",
                                "Comment icon with checkmark overlay when comment provided",
                            ],
                            "Summary": "Dual feedback system: sentiment tracking + LangSmith integration with localStorage persistence.",
                        },
                        "Summary": "Comprehensive feedback capturing thumbs up/down and optional comments with visual confirmation.",
                    },
                    "Summary": "Complete message display with markdown, datasets, SQL/PDF modals, follow-ups, and feedback.",
                },
                "InputBar": {
                    "Details": [
                        "Controlled input with currentMessage state",
                        "Rounded-full bg-[#F9F9F5] styling",
                        "Send button with rotate-45 arrow icon",
                        "Disabled during loading",
                    ],
                    "Summary": "Simple message input component with send button (legacy, not used in main chat).",
                },
                "Summary": "Chat-specific components for message display, input, and user feedback.",
            },
            "Data components": {
                "DatasetsTable (Catalog)": {
                    "Details": [
                        "Pagination: page + page_size=10",
                        "Client-side filtering with removeDiacritics() multi-word search",
                        "Backend pagination when no filter, client filtering when filter active",
                        "LocalStorage: czsu-catalog-page, czsu-catalog-filter",
                        "onRowClick callback navigates to /data",
                        "dataset-code-badge styling for selection codes",
                    ],
                    "Summary": "Paginated catalog table with hybrid filtering and localStorage state persistence.",
                },
                "DataTableView": {
                    "Features": {
                        "Details": [
                            "Auto-complete search suggestions from /data-tables",
                            "* prefix for code-only search",
                            "Click suggestion or press Enter to load table",
                            "Column filters with numeric operators for 'value' column",
                            "Sortable columns (click header to cycle asc/desc/none)",
                            "Visual sort indicators (▲ ▼ neutral)",
                            "Click table code badge to prefill catalog filter",
                        ],
                        "Summary": "Advanced table viewer with auto-complete, filtering, sorting, and cross-navigation.",
                    },
                    "State sync": {
                        "Details": [
                            "Props: search, setSearch, selectedTable, setSelectedTable, etc.",
                            "All state lifted to parent for localStorage persistence",
                            "pendingTableSearch from URL param auto-loads table",
                        ],
                        "Summary": "Lifted state pattern ensuring localStorage persistence across navigation.",
                    },
                    "Summary": "Feature-rich data table component with intelligent search, filters, sorting, and state management.",
                },
                "Summary": "Data exploration components for catalog browsing and detailed table viewing.",
            },
            "Utility components": {
                "LoadingSpinner": {
                    "Details": [
                        "Sizes: sm, md, lg with different w-h classes",
                        "Blue spinner with transparent top border (border-t-transparent)",
                        "Optional text label with size-matched font",
                        "role='status' and aria-label for accessibility",
                    ],
                    "Summary": "Reusable loading indicator with size variants and accessibility support.",
                },
                "Modal": {
                    "Details": [
                        "Fixed overlay with bg-black bg-opacity-40",
                        "Centered white rounded-lg card with max-w-2xl",
                        "Close button (×) in top-right",
                        "Escape key to close (useEffect listener)",
                        "z-50 to overlay everything",
                    ],
                    "Summary": "Generic modal component for SQL/PDF displays with keyboard and click-to-close.",
                },
                "Summary": "Small reusable utilities for loading states and modal overlays.",
            },
            "Summary": "Comprehensive component library covering layout, chat, data, and utilities.",
        },
        "State Management": {
            "Context Providers": {
                "ChatCacheContext": {
                    "State": {
                        "Details": [
                            "threads: ChatThreadMeta[] - all loaded threads",
                            "messages: {[threadId]: ChatMessage[]} - messages per thread",
                            "runIds: {[threadId]: {run_id, prompt, timestamp}[]} - run IDs per thread",
                            "sentiments: {[threadId]: {[runId]: boolean}} - thumbs per run",
                            "activeThreadId: string | null - currently selected thread",
                            "userEmail: string | null - authenticated user",
                            "Pagination: threadsPage, threadsHasMore, threadsLoading, totalThreadsCount",
                        ],
                        "Summary": "Centralized cache for threads, messages, run IDs, sentiments, and pagination state.",
                    },
                    "localStorage persistence": {
                        "Details": [
                            "CACHE_KEY: 'czsu-chat-cache' - main cache object",
                            "ACTIVE_THREAD_KEY: 'czsu-last-active-chat' - separate for quick restore",
                            "CACHE_DURATION: 48 hours - staleness threshold",
                            "saveToStorage() on every state change (debounced by hasBeenHydrated)",
                            "loadFromStorage() on mount for instant UI",
                            "Cross-tab sync via 'storage' event listener",
                        ],
                        "Summary": "48-hour localStorage cache with automatic save/load and cross-tab synchronization.",
                    },
                    "Pagination": {
                        "Details": [
                            "loadInitialThreads(): fetches /chat-threads?page=1&limit=10",
                            "loadMoreThreads(): fetches next page, appends to threads[]",
                            "resetPagination(): clears threads, resets page to 1",
                            "Triggers loadAllMessagesFromAPI() after loading threads",
                        ],
                        "Summary": "Server-side pagination for threads with automatic message loading.",
                    },
                    "Bulk loading": {
                        "Details": [
                            "loadAllMessagesFromAPI(): calls /chat/all-messages-for-all-threads",
                            "Returns messages, runIds, sentiments for ALL threads in one request",
                            "Replaces N individual API calls with 1 bulk call",
                            "Triggered after loadInitialThreads() and loadMoreThreads()",
                            "Logs performance: 'Loaded X messages with 1 API call instead of Y'",
                        ],
                        "Summary": "Optimized bulk loading reduces API calls and improves performance.",
                    },
                    "Page refresh detection": {
                        "Details": [
                            "performance.navigation.type === 1 (legacy) or performance.getEntriesByType('navigation')[0].type === 'reload'",
                            "F5_REFRESH_THROTTLE_KEY with 5-minute cooldown",
                            "isPageRefresh flag forces API call instead of cache on F5",
                            "Navigation uses cache, F5 forces fresh data",
                        ],
                        "Summary": "Intelligent detection distinguishes F5 refresh from navigation to optimize data loading.",
                    },
                    "Cross-tab loading": {
                        "Details": [
                            "USER_LOADING_STATE_KEY: 'czsu-user-loading-{email}'",
                            "setUserLoadingState(email, true/false) in localStorage",
                            "checkUserLoadingState(email) prevents concurrent requests across tabs",
                            "30-second expiry for stale loading states",
                        ],
                        "Summary": "Prevents race conditions when same user has multiple tabs open.",
                    },
                    "Actions": {
                        "Details": [
                            "setThreads, setMessages, setActiveThreadId - state setters",
                            "addMessage, updateMessage - message CRUD",
                            "addThread, removeThread, updateThread - thread CRUD",
                            "invalidateCache, refreshFromAPI, forceAPIRefresh - cache control",
                            "hasMessagesForThread(threadId) - check message existence",
                            "getRunIdsForThread, getSentimentsForThread, updateCachedSentiment - metadata access",
                        ],
                        "Summary": "Rich API for managing threads, messages, cache, and metadata.",
                    },
                    "Summary": "Sophisticated cache provider managing threads, messages, run IDs, sentiments with localStorage, pagination, bulk loading, page refresh detection, and cross-tab sync.",
                },
                "Summary": "ChatCacheContext is the primary state management solution, replacing Redux/Zustand.",
            },
            "Hooks": {
                "useChatCache": {
                    "Details": [
                        "Context consumer for ChatCacheContext",
                        "Returns all cache state + actions",
                        "Throws error if used outside provider",
                    ],
                    "Summary": "Hook to access chat cache from any component.",
                },
                "useSentiment": {
                    "Details": [
                        "Manages sentiment state: {[runId]: boolean | null}",
                        "loadSentiments(threadId) from ChatCacheContext",
                        "updateSentiment(runId, sentiment) with optimistic UI + /sentiment POST",
                        "getSentimentForRunId(runId) for checking current state",
                        "Clears old localStorage sentiment data (database-only approach)",
                    ],
                    "Summary": "Hook for sentiment management with database sync and optimistic updates.",
                },
                "useInfiniteScroll": {
                    "Details": [
                        "IntersectionObserver-based infinite scroll",
                        "Options: threshold (default 1.0), rootMargin (default '0px')",
                        "Returns: isLoading, error, hasMore, loadMore(), observerRef",
                        "Prevents duplicate calls with isLoadingRef flag",
                        "Auto-triggers onLoadMore() when observerRef element enters viewport",
                    ],
                    "Summary": "Reusable infinite scroll hook for pagination.",
                },
                "Summary": "Custom hooks for cache access, sentiment tracking, and infinite scroll pagination.",
            },
            "Summary": "Context-based state management with ChatCacheContext + custom hooks for specialized functionality.",
        },
        "API Integration": {
            "Configuration (lib/api.ts)": {
                "Details": [
                    "API_CONFIG.baseUrl: /api in production (vercel.json rewrites), http://localhost:8000 in dev",
                    "API_CONFIG.timeout: 600000ms (10 minutes)",
                    "createFetchOptions() adds Content-Type: application/json",
                    "createAuthFetchOptions(token) adds Authorization: Bearer {token}",
                ],
                "Summary": "Centralized API configuration with environment-aware base URL and auth helpers.",
            },
            "Fetch wrappers": {
                "apiFetch<T>": {
                    "Details": [
                        "Generic fetch with timeout using AbortSignal.timeout()",
                        "Logs request details (method, headers, body length)",
                        "Parses JSON response",
                        "Throws with status code on HTTP errors",
                        "Adds context to timeout errors",
                    ],
                    "Summary": "Basic authenticated fetch with timeout, logging, and error handling.",
                },
                "authApiFetch<T>": {
                    "Details": [
                        "Wraps apiFetch() with Authorization header",
                        "Auto-refresh on 401 Unauthorized: calls getSession() for new token",
                        "Retries original request with fresh token",
                        "Throws user-friendly errors: 'Session expired - please refresh'",
                        "Logs refresh attempts and token state",
                    ],
                    "Summary": "Enhanced fetch with automatic token refresh on 401 errors.",
                },
                "Summary": "Robust fetch utilities with timeout, auth, auto-refresh, and comprehensive logging.",
            },
            "Backend routes": {
                "Chat": {
                    "Details": [
                        "/chat-threads?page={page}&limit={limit} - paginated thread list",
                        "/chat/{thread_id} - DELETE for thread removal",
                        "/chat/all-messages-for-all-threads - bulk messages/runIds/sentiments",
                        "/chat/all-messages-for-one-thread/{thread_id} - single thread data",
                    ],
                    "Summary": "Chat thread CRUD and bulk data retrieval endpoints.",
                },
                "Analysis": {
                    "Details": [
                        "/analyze - POST {prompt, thread_id, run_id} → AnalyzeResponse",
                        "/stop-execution - POST {thread_id, run_id} for cancellation",
                    ],
                    "Summary": "Core analysis endpoints for question answering and execution control.",
                },
                "Feedback": {
                    "Details": [
                        "/feedback - POST {run_id, feedback, comment?} to LangSmith",
                        "/sentiment - POST {run_id, sentiment: boolean | null} to database",
                    ],
                    "Summary": "Dual feedback system: LangSmith scores + database sentiments.",
                },
                "Data": {
                    "Details": [
                        "/catalog?page={page}&page_size={size} - dataset catalog pagination",
                        "/data-tables - all table codes with descriptions",
                        "/data-table?table={code} - table columns + rows",
                    ],
                    "Summary": "Dataset browsing and table data retrieval endpoints.",
                },
                "Misc": {
                    "Details": [
                        "/initial-followup-prompts - default follow-up suggestions for new chats",
                    ],
                    "Summary": "Helper endpoints for UI enhancements.",
                },
                "Summary": "Comprehensive REST API covering chat, analysis, feedback, sentiment, and data exploration.",
            },
            "Summary": "Sophisticated API layer with auto-refreshing auth, timeout handling, and extensive backend integration.",
        },
        "Authentication": {
            "NextAuth setup (app/api/auth/[...nextauth]/route.ts)": {
                "Providers": {
                    "Details": [
                        "GoogleProvider with GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET",
                        "Scopes: openid email profile",
                        "access_type: offline, prompt: consent for refresh_token",
                    ],
                    "Summary": "Google OAuth provider with offline access for token refresh.",
                },
                "Token refresh": {
                    "Details": [
                        "refreshAccessToken() calls oauth2.googleapis.com/token",
                        "Uses refresh_token to get new access_token + id_token",
                        "Updates expires_at timestamp",
                        "Returns error: 'RefreshAccessTokenError' on failure",
                    ],
                    "Summary": "Automatic token refresh mechanism for long-lived sessions.",
                },
                "Callbacks": {
                    "jwt()": {
                        "Details": [
                            "On sign in: stores accessToken, refreshToken, id_token, expires_at",
                            "Adds user profile data: picture, name, email",
                            "Checks expiry and calls refreshAccessToken() if needed",
                        ],
                        "Summary": "JWT callback managing token lifecycle and refresh logic.",
                    },
                    "session()": {
                        "Details": [
                            "Adds accessToken, refreshToken, id_token to session object",
                            "Ensures user.image, user.name, user.email from token",
                        ],
                        "Summary": "Session callback exposing tokens and user info to client.",
                    },
                },
                "Summary": "Complete NextAuth configuration with Google OAuth, token refresh, and session management.",
            },
            "Client-side auth": {
                "Details": [
                    "useSession() hook from next-auth/react",
                    "signIn('google') for login trigger",
                    "signOut() with clearCacheForUserChange() cleanup",
                    "session.user: {name, email, image}",
                    "session.id_token for backend API authentication",
                ],
                "Summary": "Client hooks leveraging NextAuth session for UI and API calls.",
            },
            "Summary": "Robust authentication with Google OAuth, automatic token refresh, and comprehensive session management.",
        },
        "Styling": {
            "TailwindCSS 4 (globals.css)": {
                "CSS Variables": {
                    "Details": [
                        "--font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                        "--background: #F8FAFC (light) / #0a0a0a (dark)",
                        "--foreground: #181C3A (light) / #ededed (dark)",
                        "Table fonts: --table-font-family, --table-font-size, etc.",
                    ],
                    "Summary": "Centralized CSS variables for fonts, colors, and table styling.",
                },
                "Custom classes": {
                    "Details": [
                        ".light-blue-theme: bg-blue-100, hover:bg-blue-500 + white text",
                        ".chat-scrollbar: custom webkit-scrollbar with blue thumb",
                        ".dataset-code-badge: rounded-full mono badge for dataset codes",
                        ".main-container-unified: consistent spacing for all pages",
                        ".unified-white-block-system: white card with shadow, border, rounded corners",
                        ".table-container, .filter-section, .table-section, .pagination-section - layout utilities",
                    ],
                    "Summary": "Extensive custom utility classes for consistent UI patterns.",
                },
                "Summary": "TailwindCSS 4 with custom CSS variables and utility classes for cohesive design system.",
            },
            "Component styling patterns": {
                "Details": [
                    "rounded-2xl for cards, rounded-full for buttons/badges",
                    "shadow-2xl for elevated surfaces, shadow-md for subtle depth",
                    "bg-gradient-to-br from-blue-100 via-blue-50 to-blue-200 for backgrounds",
                    "hover: and focus: states for interactive elements",
                    "transition-all duration-200 for smooth animations",
                ],
                "Summary": "Consistent styling patterns across components using Tailwind utilities.",
            },
            "Summary": "Modern styling with TailwindCSS 4, CSS variables, and consistent custom classes.",
        },
        "Utilities": {
            "Text processing (components/utils.ts)": {
                "Details": [
                    "removeDiacritics(str): NFD normalization + diacritics removal for Czech text",
                    "Used in DatasetsTable and DataTableView for filtering",
                ],
                "Summary": "Czech-aware text normalization for search and filtering.",
            },
            "IndexedDB (components/utils.ts - chatDb)": {
                "Details": [
                    "idb library wrapper for czsu-chat-modern database",
                    "Object stores: threads (by-user index), messages (by-thread, by-user indexes)",
                    "CRUD: listThreads, getChatThread, saveThread, deleteThread",
                    "Message CRUD: listMessages, saveMessage, deleteMessage",
                    "LocalChatThreadMeta extends SharedChatThreadMeta with user + timestamps",
                ],
                "Summary": "Structured IndexedDB storage for chat data (legacy, superseded by ChatCacheContext localStorage).",
            },
            "Summary": "Utility functions for text processing and client-side database (mostly legacy).",
        },
        "Types (src/types/index.ts)": {
            "Details": [
                "ChatThreadMeta: thread_id, latest_timestamp, run_count, title, full_prompt",
                "ChatMessage: id, threadId, user, createdAt, prompt?, final_answer?, followup_prompts?, queries_and_results, datasets_used, top_chunks, sql_query, error?, isLoading?, run_id?",
                "AnalyzeRequest: prompt, thread_id, run_id?",
                "AnalyzeResponse: result, followup_prompts?, queries_and_results, datasets_used?, iteration, max_iterations, sql, run_id, top_chunks, warning?",
                "FeedbackRequest: run_id, feedback (0 or 1), comment?",
                "SentimentRequest: run_id, sentiment (boolean | null)",
                "PaginatedChatThreadsResponse: threads[], total_count, page, limit, has_more",
                "ApiConfig, ApiError",
            ],
            "Summary": "Comprehensive TypeScript interfaces ensuring type safety across frontend-backend communication.",
        },
        "Summary": "Production-ready Next.js 15 application with App Router, NextAuth Google OAuth, TailwindCSS 4, TypeScript, ChatCacheContext for state, sophisticated API integration with auto-refresh, infinite scroll pagination, markdown rendering, sentiment tracking, IndexedDB (legacy), localStorage persistence, cross-tab sync, and comprehensive error recovery.",
    }
}


def create_mindmap_graph(mindmap_dict, graph=None, parent=None, level=0):
    """Recursively create a Graphviz graph from the mindmap dictionary."""
    if graph is None:
        graph = Digraph(comment="Frontend Next.js Mindmap")
        graph.attr(rankdir="LR")  # Left to right layout for horizontal mindmap

    colors = ["lightblue", "lightgreen", "lightyellow", "lightpink", "lightcyan"]

    for key, value in mindmap_dict.items():
        node_id = f"{parent}_{key}" if parent else key
        node_id = (
            node_id.replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(".", "_")
            .replace("[", "")
            .replace("]", "")
            .replace(":", "")
        )

        # Set node color based on level
        color = colors[min(level, len(colors) - 1)]

        if isinstance(value, dict):
            # This is a branch node
            graph.node(node_id, key, shape="box", style="filled", fillcolor=color)
            if parent:
                graph.edge(parent, node_id)
            create_mindmap_graph(value, graph, node_id, level + 1)
        elif isinstance(value, list):
            # This is a leaf node with multiple items
            graph.node(node_id, key, shape="ellipse", style="filled", fillcolor=color)
            if parent:
                graph.edge(parent, node_id)
            for item in value:
                item_id = f"{node_id}_{item.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_').replace('.', '_').replace('[', '').replace(']', '').replace(':', '').replace('@', '_at_').replace(',', '').replace('&', 'and')[:50]}"
                graph.node(item_id, item, shape="plaintext")
                graph.edge(node_id, item_id)
        else:
            # Single leaf node
            graph.node(node_id, str(value), shape="plaintext")
            if parent:
                graph.edge(parent, node_id)

    return graph


def main():
    """Generate and save the mindmap visualization."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    graph = create_mindmap_graph(mindmap)

    # Save as PNG
    png_path = os.path.join(script_dir, script_name)
    graph.render(png_path, format="png", cleanup=True)
    print(f"Mindmap saved as '{png_path}.png'")

    # Also save as PDF for better quality
    pdf_path = os.path.join(script_dir, script_name)
    graph.render(pdf_path, format="pdf", cleanup=True)
    print(f"Mindmap saved as '{pdf_path}.pdf'")

    # Print text representation
    print("\nText-based Mindmap:")
    print_mindmap_text(mindmap)


def print_mindmap_text(mindmap_dict, prefix=""):
    """Print a text-based representation of the mindmap in a vertical tree format."""
    keys = list(mindmap_dict.keys())
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + key)

        value = mindmap_dict[key]
        if isinstance(value, dict):
            extension = "    " if is_last else "│   "
            print_mindmap_text(value, prefix + extension)
        elif isinstance(value, list):
            for j, item in enumerate(value):
                is_last_sub = j == len(value) - 1
                sub_connector = "└── " if is_last_sub else "├── "
                sub_extension = "    " if is_last else "│   "
                print(prefix + sub_extension + sub_connector + item)


if __name__ == "__main__":
    main()
