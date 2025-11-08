# CZSU Multi-Agent Text-to-SQL Application Architecture

## Overview

The CZSU Multi-Agent Text-to-SQL application is a modern **server-side rendered web application** built with **Next.js 15** using the **App Router** pattern, combined with a **Python FastAPI backend**. The frontend uses **React 19** with **TypeScript**, **Tailwind CSS** for styling, and **NextAuth.js** for authentication. The backend provides REST APIs with **PostgreSQL** for data storage and **ChromaDB** for vector embeddings.

## Core Architecture Components

### Frontend Application: Next.js 15 App Router with Server-Side Rendering

**Framework**: Next.js 15 with App Router
**Application Type**: Multi-page web application (not a Single Page Application)
**React Version**: React 19 (latest)
**Rendering Approach**: Server-Side Rendering (SSR) with selective Client Components
**Routing**: File-system based router with nested layouts and server components
**Styling**: Tailwind CSS with custom design system
**Authentication**: NextAuth.js integration
**State Management**: React Context with localStorage persistence

### Server-Side Rendering (SSR) Implementation

#### Server Components (Default Behavior)
- **All components are server-rendered by default** - no "use client" directive needed
- **Server-side execution** for initial page loads and static content
- **Metadata API integration** for SEO optimization in `layout.tsx`
- **Static generation** for pages without dynamic data

**Evidence in Codebase:**
- `app/layout.tsx`: Server component with metadata export and HTML structure
- `app/page.tsx`: Server component rendering welcome page content
- `app/not-found.tsx`: Server component for 404 error pages

#### Client Components (Explicit Opt-in)
- **Selective client-side hydration** using `"use client"` directive
- **Client-side interactivity** for dynamic features requiring browser APIs
- **State management** with React hooks and context

**Evidence in Codebase:**
- `app/ClientLayout.tsx`: Client component with navigation logic and auth redirects
- `app/docs/page.tsx`: Client component for Swagger UI with data fetching
- `components/AuthGuard.tsx`, `components/Header.tsx`: Client components for interactive UI

#### Serverless API Routes
- **Server-side API endpoints** in `/api` directory
- **Edge runtime execution** for API routes
- **NextAuth.js integration** for authentication endpoints

**Evidence in Codebase:**
- `app/api/test/route.ts`: Serverless function returning JSON response
- `app/api/auth/[...nextauth]/route.ts`: NextAuth.js serverless authentication
- Vercel deployment configuration routing API calls to Railway backend

#### Server Components (Default Behavior)
- **All components are server-rendered by default** - no "use client" directive needed
- **Server-side execution** for initial page loads and static content
- **Metadata API integration** for SEO optimization in `layout.tsx`
- **Static generation** for pages without dynamic data

**Evidence in Codebase:**
- `app/layout.tsx`: Server component with metadata export and HTML structure
- `app/page.tsx`: Server component rendering welcome page content
- `app/not-found.tsx`: Server component for 404 error pages

#### Client Components (Explicit Opt-in)
- **Selective client-side hydration** using `"use client"` directive
- **Client-side interactivity** for dynamic features requiring browser APIs
- **State management** with React hooks and context

**Evidence in Codebase:**
- `app/ClientLayout.tsx`: Client component with navigation logic and auth redirects
- `app/docs/page.tsx`: Client component for Swagger UI with data fetching
- `components/AuthGuard.tsx`, `components/Header.tsx`: Client components for interactive UI

#### Serverless API Routes
- **Server-side API endpoints** in `/api` directory
- **Edge runtime execution** for API routes
- **NextAuth.js integration** for authentication endpoints

**Evidence in Codebase:**
- `app/api/test/route.ts`: Serverless function returning JSON response
- `app/api/auth/[...nextauth]/route.ts`: NextAuth.js serverless authentication
- Vercel deployment configuration routing API calls to Railway backend

### Component Architecture

#### Server Components
- **Default rendering mode** for static pages and layouts
- **Server-side execution** for layout and simple pages
- Used in: Root layout (`layout.tsx`), Home page (`page.tsx`), 404 page (`not-found.tsx`)

#### Client Components
- **Explicit opt-in** using `"use client"` directive
- **Client-side interactivity** for dynamic features
- Used in: Chat interface, data tables, authentication flows, Swagger UI
- **State management** with React hooks and context

### API Architecture

#### Next.js API Routes (Serverless)
- **Server-side execution** in `/api` directory
- **NextAuth.js integration** for authentication
- **Simple utility endpoints** (test routes, placeholder images)

#### Backend Services
- **Primary Backend**: Python FastAPI application
- **Database**: PostgreSQL with connection pooling
- **Vector Database**: ChromaDB for embedding storage
- **External APIs**: CZSU data integration

## Data Flow Architecture

### Request Flow
1. **Server-Side Rendering**: Initial page rendered on server with static content
2. **Client Hydration**: React components hydrate on client for interactivity
3. **API Calls**: Client components make authenticated requests to backend APIs
4. **Data Processing**: FastAPI processes requests with database/vector operations
5. **Response**: JSON data returned to frontend
6. **State Updates**: React context updates local state and localStorage

### State Management
- **Client State**: React hooks (useState, useEffect) for component state
- **Global State**: React Context for cross-component data sharing
- **Persistent State**: localStorage for caching chat threads and messages
- **Authentication State**: NextAuth.js session management

### Caching Strategy
- **Client Caching**: localStorage for chat data persistence
- **Context Caching**: In-memory caching of active chat threads
- **Database Caching**: PostgreSQL query optimization

## Security Architecture

### Authentication & Authorization
- **NextAuth.js**: JWT-based authentication with external providers
- **Session Management**: Secure token handling and refresh
- **Route Protection**: Client-side route guards with AuthGuard component
- **API Security**: Authentication middleware on backend endpoints

### Data Protection
- **Input Validation**: Client and server-side validation
- **SQL Injection Prevention**: Parameterized queries in FastAPI
- **CORS Configuration**: Proper cross-origin request handling

## Deployment Architecture

### Infrastructure
- **Frontend**: Vercel deployment with automatic SSR builds
- **Backend**: Railway cloud platform
- **Database**: Managed PostgreSQL service
- **Vector Storage**: ChromaDB with persistent storage

### Build Process
- **Frontend**: Next.js production build with SSR compilation
- **Backend**: Python application with dependency management
- **Environment**: Separate environments for development and production

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend Framework** | Next.js 15 | React framework with App Router |
| **Application Type** | Multi-page Web App | Server-side rendered with selective client hydration |
| **React Architecture** | Server Components + Client Components | SSR by default with selective client hydration |
| **Rendering Approach** | Server-Side Rendering (SSR) | HTML generated on server, enhanced with client interactivity |
| **Styling** | Tailwind CSS | Utility-first CSS framework |
| **Authentication** | NextAuth.js | Authentication library |
| **Backend Framework** | FastAPI | High-performance Python API |
| **Database** | PostgreSQL | Primary data storage |
| **Vector Database** | ChromaDB | Embedding storage |
| **Deployment** | Vercel/Railway | Cloud hosting platforms |
| **State Management** | React Context | Client-side state |
| **API Communication** | Fetch API | HTTP client |

## Architecture Benefits

### Performance
- **Fast Initial Loads**: Server-rendered layouts and static content
- **Rich Interactivity**: Client components for dynamic features
- **Optimized Bundles**: Code splitting for different routes
- **Efficient Caching**: localStorage persistence for chat data

### Developer Experience
- **Type Safety**: TypeScript throughout the frontend
- **Modern React**: Latest React 19 features
- **Unified Development**: Single repository for full-stack development
- **Hot Reloading**: Fast development iteration in Next.js

### User Experience
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS
- **Progressive Enhancement**: Works with JavaScript disabled for basic features
- **Persistent Sessions**: Chat history preserved across browser sessions
- **Real-time Updates**: Live chat interface with streaming responses

### Maintainability
- **Modular Architecture**: Clear separation between frontend and backend
- **Standard Patterns**: RESTful API design and React best practices
- **Comprehensive Testing**: Unit tests for backend utilities
- **Documentation**: API documentation with Swagger UI

This architecture provides a solid foundation for a modern **server-side rendered web application**, balancing development velocity with production reliability.