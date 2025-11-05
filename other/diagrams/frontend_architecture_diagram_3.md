# Frontend Architecture Diagram - Version 3 (Component-Interaction Focused)

```mermaid
graph TB
    %% User Entry Point
    User["<div style='font-size:40px'>👤</div><div style='font-size:10px'>User</div>"]

    %% Core Application Shell
    subgraph AppShell["🏛️ Application Shell"]
        RootLayout["<div style='font-size:30px'>📐</div><div style='font-size:9px'>RootLayout<br/>(Server Component)</div>"]
        ClientLayout["<div style='font-size:30px'>🖼️</div><div style='font-size:9px'>ClientLayout<br/>(Route Protection)</div>"]
        Header["<div style='font-size:30px'>🎯</div><div style='font-size:9px'>Header<br/>(Navigation)</div>"]
    end

    %% State Management Core
    subgraph StateCore["💾 State Management"]
        ChatCache["<div style='font-size:30px'>🗂️</div><div style='font-size:9px'>ChatCacheContext<br/>• Threads<br/>• Messages<br/>• Pagination<br/>• LocalStorage Sync</div>"]
        SessionProvider["<div style='font-size:30px'>👤</div><div style='font-size:9px'>SessionProvider<br/>• Auth State<br/>• User Info<br/>• Token Management</div>"]
    end

    %% Feature Components
    subgraph Features["🎯 Feature Components"]
        direction LR
        
        subgraph ChatFeature["💬 Chat Feature"]
            ChatPage["<div style='font-size:20px'>📄</div><div style='font-size:8px'>ChatPage</div>"]
            Sidebar["<div style='font-size:20px'>📋</div><div style='font-size:8px'>Thread Sidebar</div>"]
            MessageArea["<div style='font-size:20px'>💭</div><div style='font-size:8px'>MessageArea</div>"]
            InputBar["<div style='font-size:20px'>⌨️</div><div style='font-size:8px'>InputBar</div>"]
        end
        
        subgraph CatalogFeature["📚 Catalog Feature"]
            CatalogPage["<div style='font-size:20px'>📄</div><div style='font-size:8px'>CatalogPage</div>"]
            DatasetsTable["<div style='font-size:20px'>📊</div><div style='font-size:8px'>DatasetsTable</div>"]
        end
        
        subgraph DataFeature["📊 Data Feature"]
            DataPage["<div style='font-size:20px'>📄</div><div style='font-size:8px'>DataPage</div>"]
            DataTableView["<div style='font-size:20px'>🔍</div><div style='font-size:8px'>DataTableView<br/>+ Filters</div>"]
        end
    end

    %% API Layer
    subgraph APILayer["🌐 API Communication"]
        APIUtils["<div style='font-size:30px'>🔧</div><div style='font-size:9px'>API Utilities<br/>authApiFetch</div>"]
        TokenRefresh["<div style='font-size:30px'>🔄</div><div style='font-size:9px'>Token Refresh<br/>Auto Retry on 401</div>"]
    end

    %% Authentication
    subgraph AuthSystem["🔐 Authentication"]
        NextAuth["<div style='font-size:30px'>🎫</div><div style='font-size:9px'>NextAuth<br/>OAuth Handler</div>"]
        AuthGuard["<div style='font-size:30px'>🛡️</div><div style='font-size:9px'>AuthGuard<br/>Route Protection</div>"]
    end

    %% Storage
    subgraph Storage["💽 Client Storage"]
        LocalStorage["<div style='font-size:30px'>📦</div><div style='font-size:9px'>LocalStorage<br/>• Chat Cache<br/>• User Prefs<br/>• Cross-tab Sync</div>"]
        IndexedDB["<div style='font-size:30px'>🗄️</div><div style='font-size:9px'>IndexedDB<br/>Large Datasets</div>"]
    end

    %% External Systems
    Backend["<div style='font-size:40px'>🚀</div><div style='font-size:10px'>FastAPI Backend<br/>Railway</div>"]
    GoogleOAuth["<div style='font-size:40px'>🔐</div><div style='font-size:10px'>Google OAuth</div>"]

    %% Primary User Flow
    User -->|"Navigates to"| RootLayout
    RootLayout --> ClientLayout
    ClientLayout --> Header
    ClientLayout -->|"Wraps"| Features

    %% State Provider Flow
    ClientLayout -->|"Provides"| ChatCache
    ClientLayout -->|"Provides"| SessionProvider

    %% Feature Component Internal Flows
    ChatPage --> Sidebar
    ChatPage --> MessageArea
    ChatPage --> InputBar
    
    CatalogPage --> DatasetsTable
    DataPage --> DataTableView

    %% State Consumption
    ChatPage -->|"Reads/Writes"| ChatCache
    Sidebar -->|"Reads"| ChatCache
    MessageArea -->|"Reads"| ChatCache
    InputBar -->|"Writes"| ChatCache
    
    Header -->|"Reads"| SessionProvider
    ChatPage -->|"Reads"| SessionProvider
    CatalogPage -->|"Reads"| SessionProvider
    DataPage -->|"Reads"| SessionProvider

    %% Authentication Flow
    ClientLayout --> AuthGuard
    AuthGuard -->|"Checks"| SessionProvider
    SessionProvider -->|"Manages"| NextAuth
    NextAuth -->|"OAuth Flow"| GoogleOAuth

    %% API Communication Flow
    InputBar -->|"Send Message"| APIUtils
    DatasetsTable -->|"Fetch Datasets"| APIUtils
    DataTableView -->|"Fetch Data"| APIUtils
    
    APIUtils -->|"HTTP Request"| Backend
    APIUtils -->|"On 401"| TokenRefresh
    TokenRefresh -->|"Refresh Token"| NextAuth
    TokenRefresh -->|"Retry Request"| Backend

    %% Storage Flow
    ChatCache <-->|"Persist/Restore"| LocalStorage
    DataTableView <-->|"Store Large Data"| IndexedDB
    LocalStorage -.->|"Cross-tab Events"| ChatCache

    %% Data Response Flow
    Backend -->|"Response"| APIUtils
    APIUtils -->|"Update State"| ChatCache
    ChatCache -->|"Trigger Re-render"| Features

    %% Styling
    classDef userStyle fill:#fef3c7,stroke:#f59e0b,stroke-width:4px,color:#000
    classDef shellStyle fill:#dbeafe,stroke:#3b82f6,stroke-width:3px,color:#000
    classDef stateStyle fill:#fce7f3,stroke:#ec4899,stroke-width:3px,color:#000
    classDef featureStyle fill:#e0e7ff,stroke:#6366f1,stroke-width:3px,color:#000
    classDef apiStyle fill:#cffafe,stroke:#06b6d4,stroke-width:3px,color:#000
    classDef authStyle fill:#fef2f2,stroke:#ef4444,stroke-width:3px,color:#000
    classDef storageStyle fill:#d1fae5,stroke:#10b981,stroke-width:3px,color:#000
    classDef externalStyle fill:#1e293b,stroke:#0ea5e9,stroke-width:3px,color:#fff

    class User userStyle
    class AppShell shellStyle
    class StateCore stateStyle
    class Features,ChatFeature,CatalogFeature,DataFeature featureStyle
    class APILayer apiStyle
    class AuthSystem authStyle
    class Storage storageStyle
    class Backend,GoogleOAuth externalStyle
```

## Architecture Overview

### 🎯 Core Architectural Components

#### 1. **Application Shell** (Entry Point)
- **RootLayout**: Server-side rendered HTML structure
- **ClientLayout**: Client-side route protection and state providers
- **Header**: Global navigation component

#### 2. **State Management** (Single Source of Truth)
- **ChatCacheContext**: 
  - Manages all chat-related data (threads, messages, pagination)
  - Persists to LocalStorage with cross-tab synchronization
  - Provides optimistic updates for instant UI feedback
- **SessionProvider**:
  - Manages authentication state
  - Provides user information to all components
  - Handles token lifecycle

#### 3. **Feature Components** (Business Logic)
- **Chat Feature**: 
  - Sidebar displays thread list from ChatCache
  - MessageArea renders messages from ChatCache
  - InputBar sends messages via API and updates ChatCache
- **Catalog Feature**:
  - Displays browsable dataset catalog
  - Fetches data via API utilities
- **Data Feature**:
  - Table view with filtering capabilities
  - Uses IndexedDB for large dataset storage

#### 4. **API Communication** (Backend Integration)
- **API Utilities**: Centralized HTTP request handling
- **Token Refresh**: Automatic retry mechanism on authentication failures
- Handles all backend communication with error recovery

#### 5. **Authentication System** (Security)
- **NextAuth**: OAuth 2.0 flow handler
- **AuthGuard**: Protects routes from unauthorized access
- Integrated with SessionProvider for state management

#### 6. **Client Storage** (Performance)
- **LocalStorage**: Fast access for cache and preferences
- **IndexedDB**: Large dataset storage without memory pressure
- Cross-tab synchronization for consistent state

### 🔄 Key Interaction Patterns

#### Message Send Flow:
```
User types message
    ↓
InputBar captures input
    ↓
InputBar calls APIUtils.authApiFetch('/analyze')
    ↓
APIUtils sends to Backend with auth token
    ↓ (if 401)
TokenRefresh gets new token → Retry request
    ↓
Backend returns response
    ↓
ChatCache updates with new message
    ↓
MessageArea re-renders with new data
```

#### Authentication Flow:
```
User navigates to protected route
    ↓
AuthGuard checks SessionProvider
    ↓ (if not authenticated)
Redirect to NextAuth login
    ↓
NextAuth redirects to Google OAuth
    ↓
Google returns with tokens
    ↓
NextAuth creates session
    ↓
SessionProvider updates state
    ↓
User redirected to protected route
```

#### Cache Persistence Flow:
```
ChatCache state changes
    ↓
useEffect triggers save to LocalStorage
    ↓
Other browser tabs receive storage event
    ↓
Other tabs' ChatCache syncs with new data
    ↓
All tabs show consistent state
```

### 🏗️ Architectural Principles

1. **Unidirectional Data Flow**: State flows down, events flow up
2. **Single Source of Truth**: ChatCacheContext is authoritative for chat data
3. **Separation of Concerns**: Components, state, API, and auth are independent
4. **Error Recovery**: Automatic token refresh and retry on failures
5. **Performance**: LocalStorage + IndexedDB for fast access and large data
6. **Consistency**: Cross-tab synchronization ensures consistent state

### 📊 Component Communication Matrix

| Component | Reads From | Writes To | Communicates With |
|-----------|-----------|-----------|-------------------|
| InputBar | SessionProvider | ChatCache, API | Backend |
| MessageArea | ChatCache | - | - |
| Sidebar | ChatCache | ChatCache | - |
| ChatCache | LocalStorage | LocalStorage | - |
| APIUtils | SessionProvider | - | Backend, TokenRefresh |
| AuthGuard | SessionProvider | - | NextAuth |

---

**Version 3 Focus**: Component-based architecture showing actual components, their relationships, data flow patterns, and interaction mechanisms. Emphasizes HOW components work together rather than WHAT layers exist.

