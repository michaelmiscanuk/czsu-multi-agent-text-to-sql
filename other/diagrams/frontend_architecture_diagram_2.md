# Frontend Architecture Diagram - Version 2 (Simplified)

```mermaid
graph TB
    %% Core Framework
    subgraph Framework["⚡ Core Framework"]
        NextJS["<div style='font-size:40px'>▲</div><div style='font-size:10px'>Next.js 15.3<br/>React 19 + TypeScript</div>"]
    end

    %% Application Layer
    subgraph Application["🌐 Application Layer"]
        Layout["<div style='font-size:40px'>🎨</div><div style='font-size:10px'>Layout System<br/>Header + Navigation</div>"]
        
        subgraph CorePages["Core Pages"]
            Chat["<div style='font-size:24px'>💬</div><div style='font-size:9px'>Chat Interface<br/>AI Conversations</div>"]
            Catalog["<div style='font-size:24px'>📚</div><div style='font-size:9px'>Dataset Catalog<br/>Browse Tables</div>"]
            Data["<div style='font-size:24px'>📊</div><div style='font-size:9px'>Data Explorer<br/>View & Filter</div>"]
        end
        
        Components["<div style='font-size:40px'>🧩</div><div style='font-size:10px'>UI Components<br/>Reusable Elements</div>"]
    end

    %% State Management
    subgraph State["💾 State Management"]
        ChatCache["<div style='font-size:40px'>🗂️</div><div style='font-size:10px'>ChatCacheContext<br/>Threads + Messages + Pagination</div>"]
        Session["<div style='font-size:40px'>👤</div><div style='font-size:10px'>Session State<br/>NextAuth Provider</div>"]
        ClientStorage["<div style='font-size:40px'>📦</div><div style='font-size:10px'>Client Storage<br/>LocalStorage + IndexedDB</div>"]
    end

    %% Authentication
    subgraph Auth["🔐 Authentication"]
        NextAuth["<div style='font-size:40px'>🎫</div><div style='font-size:10px'>NextAuth.js<br/>OAuth + Token Management</div>"]
    end

    %% API Layer
    subgraph API["🌐 API Communication"]
        APIUtils["<div style='font-size:40px'>🔧</div><div style='font-size:10px'>API Utilities<br/>authApiFetch + Token Refresh</div>"]
    end

    %% External Services
    subgraph External["🔌 External Services"]
        Backend["<div style='font-size:40px'>🚀</div><div style='font-size:10px'>FastAPI Backend<br/>AI Processing + Data</div>"]
        Google["<div style='font-size:40px'>🔐</div><div style='font-size:10px'>Google OAuth 2.0<br/>Authentication Provider</div>"]
    end

    %% Primary Data Flow
    Framework --> Application
    Application --> Layout
    Application --> CorePages
    Application --> Components
    
    CorePages --> Chat
    CorePages --> Catalog
    CorePages --> Data
    
    %% State Management Flow
    Layout --> Session
    CorePages --> ChatCache
    CorePages --> Session
    Components --> Session
    
    ChatCache --> ClientStorage
    Session --> NextAuth
    
    %% Authentication Flow
    Application --> NextAuth
    NextAuth --> Google
    
    %% API Communication Flow
    Chat --> APIUtils
    Catalog --> APIUtils
    Data --> APIUtils
    
    APIUtils --> Session
    APIUtils --> Backend
    
    %% Cross-cutting Concerns
    APIUtils -.->|"401 Retry + Token Refresh"| NextAuth
    ChatCache -.->|"Cross-tab Sync"| ClientStorage

    %% Styling for subgraphs
    classDef frameworkStyle fill:#1e293b,stroke:#0ea5e9,stroke-width:3px,color:#fff
    classDef appStyle fill:#dbeafe,stroke:#3b82f6,stroke-width:3px,color:#000
    classDef stateStyle fill:#fce7f3,stroke:#ec4899,stroke-width:3px,color:#000
    classDef authStyle fill:#fef2f2,stroke:#ef4444,stroke-width:3px,color:#000
    classDef apiStyle fill:#cffafe,stroke:#06b6d4,stroke-width:3px,color:#000
    classDef externalStyle fill:#ffedd5,stroke:#f97316,stroke-width:3px,color:#000

    class Framework frameworkStyle
    class Application appStyle
    class State stateStyle
    class Auth authStyle
    class API apiStyle
    class External externalStyle
```

## Simplified Architecture Overview

### 🎯 Core Layers (6 Total)

1. **⚡ Core Framework**: Next.js 15 with React 19 and TypeScript
2. **🌐 Application Layer**: Pages, Layout, and UI Components
3. **💾 State Management**: Chat cache, Session, and Client storage
4. **🔐 Authentication**: NextAuth with Google OAuth
5. **🌐 API Communication**: Centralized API utilities with token refresh
6. **🔌 External Services**: Backend and Google OAuth provider

### 🔄 Key Data Flows

#### 1. User Interaction Flow
```
User → Page Component → State Management → API Utils → Backend
```

#### 2. Authentication Flow
```
User → NextAuth → Google OAuth → Session State → Protected Pages
```

#### 3. Chat Flow
```
Chat Page → ChatCacheContext → API Utils → Backend → Cache Update → UI Render
```

### 💡 Key Architectural Decisions

1. **Centralized State**: ChatCacheContext manages all chat-related data with persistence
2. **Token Management**: Automatic token refresh on 401 responses
3. **Client-Side Caching**: LocalStorage for fast access, IndexedDB for large data
4. **Protected Routes**: NextAuth guards all sensitive pages
5. **Pagination**: Incremental loading for better performance

### 📊 Technology Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 15, React 19, TypeScript |
| Auth | NextAuth, Google OAuth 2.0 |
| State | React Context API, LocalStorage |
| Styling | Tailwind CSS 4 |
| API | Fetch with custom wrapper |

### 🚀 Deployment

```
Vercel Edge Network
    ↓
Next.js Frontend
    ↓
FastAPI Backend (Railway)
```

---

**Version 2 Changes**: Simplified from 10 layers to 6 core layers, removed implementation details, focused on essential architecture and data flow patterns.

