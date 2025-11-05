# Frontend Architecture Diagram - Version 4 (UI-Based View)

## Application UI Architecture

```mermaid
graph TB
    %% Application Container
    subgraph App["🖥️ CZSU Data Explorer and Chatbot"]
        
        %% Global Header
        subgraph GlobalHeader["📋 Global Header (All Pages)"]
            Logo["<div style='font-size:24px'>🏛️</div><div style='font-size:8px'>App Logo<br/>CZSU Data Explorer</div>"]
            Nav["<div style='font-size:24px'>🧭</div><div style='font-size:8px'>Navigation Menu<br/>Home | Chat | Catalog | Data | Contacts</div>"]
            UserSection["<div style='font-size:24px'>👤</div><div style='font-size:8px'>User Section<br/>Avatar + Name + Sign Out</div>"]
        end
        
        %% Page Views
        subgraph PageViews["📱 Page Views (Content Area)"]
            
            %% Chat Page Layout
            subgraph ChatView["💬 Chat Page (/chat)"]
                direction LR
                
                subgraph ChatSidebar["📂 Left Sidebar"]
                    NewChatBtn["<div style='font-size:20px'>➕</div><div style='font-size:7px'>+ New Chat Button</div>"]
                    ThreadList["<div style='font-size:20px'>📜</div><div style='font-size:7px'>Thread List<br/>• New Chat<br/>• Tell me about...<br/>• What are migration...<br/>• Show me tourism...<br/>(Scrollable with ×)</div>"]
                end
                
                subgraph ChatMain["💭 Main Chat Area"]
                    ChatHeader["<div style='font-size:20px'>💬</div><div style='font-size:7px'>Chat Icon<br/>Start a conversation</div>"]
                    Messages["<div style='font-size:20px'>📝</div><div style='font-size:7px'>Message Area<br/>(User & AI messages)</div>"]
                    Prompts["<div style='font-size:20px'>💡</div><div style='font-size:7px'>Suggested Prompts<br/>• What are trends...<br/>• Show me employment...<br/>• What is unemployment...<br/>• Show me industrial...</div>"]
                    Input["<div style='font-size:20px'>⌨️</div><div style='font-size:7px'>Input Box<br/>Type your message...<br/>+ Send Button</div>"]
                end
                
                ChatSidebar --> ChatMain
            end
            
            %% Catalog Page Layout
            subgraph CatalogView["📚 Catalog Page (/catalog)"]
                direction TB
                
                CatalogSearch["<div style='font-size:20px'>🔍</div><div style='font-size:7px'>Search Bar<br/>Filter by keyword...</div>"]
                RecordCount["<div style='font-size:20px'>📊</div><div style='font-size:7px'>Record Counter<br/>1165 records</div>"]
                
                subgraph CatalogTable["📋 Dataset Table"]
                    TableHeaders["<div style='font-size:16px'>📑</div><div style='font-size:7px'>Column Headers<br/>Selection Code | Extended Description</div>"]
                    TableRows["<div style='font-size:16px'>📄</div><div style='font-size:7px'>Dataset Rows<br/>Clickable entries with codes<br/>(e.g., CEN00282DT01)</div>"]
                end
                
                Pagination["<div style='font-size:20px'>◀️▶️</div><div style='font-size:7px'>Pagination<br/>Previous | Page 1 of 117 | Next</div>"]
                
                CatalogSearch --> RecordCount
                RecordCount --> CatalogTable
                CatalogTable --> Pagination
            end
            
            %% Data Page Layout
            subgraph DataView["📊 Data Page (/data)"]
                direction TB
                
                DataSearch["<div style='font-size:20px'>🔎</div><div style='font-size:7px'>Table Search<br/>Dataset code lookup<br/>(e.g., STAV799BT1)</div>"]
                DataInfo["<div style='font-size:20px'>ℹ️</div><div style='font-size:7px'>Table Info<br/>Starting with * searches codes<br/>1163 tables</div>"]
                
                subgraph DataTable["📊 Data Table with Filters"]
                    ColHeaders["<div style='font-size:16px'>📑</div><div style='font-size:7px'>Sortable Columns<br/>Ukazatel | Plocha bytu | ČR,kraje,okresy | Roky | value</div>"]
                    ColFilters["<div style='font-size:16px'>🔍</div><div style='font-size:7px'>Column Filters<br/>Filter... (per column)</div>"]
                    DataRows["<div style='font-size:16px'>📊</div><div style='font-size:7px'>Data Rows<br/>Průměrná plocha | Užitná plocha | Česko | 2022 | 67.01...</div>"]
                    ValueFilter["<div style='font-size:16px'>⚖️</div><div style='font-size:7px'>Value Range Filter<br/>e.g. > 10000, <= 500</div>"]
                end
                
                DataSearch --> DataInfo
                DataInfo --> DataTable
                DataTable --> ColHeaders
                ColHeaders --> ColFilters
                ColFilters --> DataRows
                DataRows --> ValueFilter
            end
            
            %% Home Page
            HomePage["<div style='font-size:40px'>🏠</div><div style='font-size:9px'>Home Page (/)<br/>Welcome message<br/>+ Links to API & PDF</div>"]
            
            %% Other Pages
            ContactsPage["<div style='font-size:40px'>📧</div><div style='font-size:9px'>Contacts Page<br/>(/contacts)</div>"]
        end
    end

    %% Navigation Flow
    GlobalHeader --> PageViews
    Nav -->|"Navigate"| HomePage
    Nav -->|"Navigate"| ChatView
    Nav -->|"Navigate"| CatalogView
    Nav -->|"Navigate"| DataView
    Nav -->|"Navigate"| ContactsPage
    
    %% User Interaction Flows
    ThreadList -.->|"Click thread"| Messages
    Messages -.->|"View conversation"| Input
    Input -.->|"Submit"| Messages
    
    TableRows -.->|"Click dataset"| DataView
    CatalogSearch -.->|"Filter"| TableRows
    
    DataSearch -.->|"Search table"| DataTable
    ColFilters -.->|"Filter columns"| DataRows

    %% Styling
    classDef headerStyle fill:#ffffff,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef chatStyle fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef catalogStyle fill:#fce7f3,stroke:#ec4899,stroke-width:2px,color:#000
    classDef dataStyle fill:#d1fae5,stroke:#10b981,stroke-width:2px,color:#000
    classDef pageStyle fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#000
    
    class GlobalHeader headerStyle
    class ChatView,ChatSidebar,ChatMain chatStyle
    class CatalogView,CatalogTable catalogStyle
    class DataView,DataTable dataStyle
    class HomePage,ContactsPage pageStyle
```

## UI Architecture Breakdown

### 🎨 Visual Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ 📋 GLOBAL HEADER (Fixed, All Pages)                         │
│ ┌──────────┬─────────────────────────────┬────────────────┐ │
│ │ 🏛️ Logo  │ 🧭 Home│Chat│Catalog│Data   │ 👤 User Menu  │ │
│ └──────────┴─────────────────────────────┴────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ 📱 PAGE CONTENT (Changes based on route)                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 💬 Chat Page UI Structure

```
┌─────────────────────────────────────────────────────┐
│ HEADER: CZSU Data Explorer and Chatbot             │
├──────────────┬──────────────────────────────────────┤
│ 📂 SIDEBAR   │ 💭 MAIN CHAT AREA                   │
│              │                                      │
│ ➕ New Chat  │     💬 Start a conversation         │
│              │     Ask me about your data...       │
│ 📜 Threads:  │                                      │
│ • New Chat   │     📝 [Message bubbles appear here]│
│ • Tell me... │                                      │
│ • What are...│     💡 Suggested prompts:           │
│ • Show me... │     • What are trends...            │
│   (scroll)   │     • Show me employment...         │
│              │                                      │
│              │     ⌨️ [Type message...] [Send]     │
└──────────────┴──────────────────────────────────────┘
```

### 📚 Catalog Page UI Structure

```
┌─────────────────────────────────────────────────────┐
│ HEADER: CZSU Data Explorer and Chatbot             │
├─────────────────────────────────────────────────────┤
│ 🔍 [Filter by keyword...]        📊 1165 records   │
├─────────────────────────────────────────────────────┤
│ 📋 DATASET TABLE                                    │
│ ┌─────────────┬───────────────────────────────────┐│
│ │Selection    │Extended Description               ││
│ │Code         │                                   ││
│ ├─────────────┼───────────────────────────────────┤│
│ │CEN00282DT01 │This dataset focuses on the...    ││
│ │             │Construction work price index...   ││
│ │             │[Full description with details]    ││
│ └─────────────┴───────────────────────────────────┘│
│                                                     │
│ ◀️ Previous   Page 1 of 117   Next ▶️              │
└─────────────────────────────────────────────────────┘
```

### 📊 Data Page UI Structure

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER: CZSU Data Explorer and Chatbot                     │
├─────────────────────────────────────────────────────────────┤
│ 🔎 [STAV799BT1]  ⓘ Starting with * searches codes (1163)  │
├─────────────────────────────────────────────────────────────┤
│ 📊 DATA TABLE WITH FILTERS                                 │
│ ┌──────────┬──────────┬─────────┬──────┬───────────────────┐│
│ │Ukazatel ⇅│Plocha ⇅  │ČR,kraje⇅│Roky ⇅│value ⇅           ││
│ ├──────────┼──────────┼─────────┼──────┼───────────────────┤│
│ │[Filter]  │[Filter]  │[Filter] │[Flt] │[Filter e.g.>10000]││
│ ├──────────┼──────────┼─────────┼──────┼───────────────────┤│
│ │Průměrná  │Užitná    │Česko    │2022  │67.0180243610505  ││
│ │plocha... │plocha    │         │      │                   ││
│ │Průměrná  │Užitná    │Česko    │2021  │66.1394331541055  ││
│ │plocha... │plocha    │         │      │                   ││
│ │...       │...       │...      │...   │...                ││
│ └──────────┴──────────┴─────────┴──────┴───────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key UI Features by Page

### Chat Page
- **Left Sidebar (Fixed)**: Thread management with scrollable list
- **Main Area**: 
  - Empty state: Welcome message + suggested prompts
  - Active state: Message history (user/AI alternating)
  - Bottom: Fixed input bar with Send button
- **Interactions**: 
  - Click thread → load messages
  - Click prompt → auto-fill input
  - Type & send → new message appears

### Catalog Page
- **Search Bar (Top)**: Filter datasets by keyword
- **Table (Main)**: 
  - Two columns: Selection Code + Extended Description
  - Clickable rows
  - Expandable descriptions
- **Pagination (Bottom)**: Navigate through 117 pages
- **Interactions**:
  - Type in search → filter results
  - Click row → navigate to Data page with that table

### Data Page
- **Search Bar (Top)**: Lookup specific table codes
- **Table (Main)**:
  - Multiple columns (indicator, area, region, year, value)
  - Sortable headers (⇅)
  - Filter inputs below each header
  - Value range filters (e.g., >10000, <=500)
- **Interactions**:
  - Type in search → load table
  - Click column header → sort
  - Type in filter → filter rows
  - Multiple filters combine (AND logic)

## 🎨 Design System

### Colors
- **Header Background**: White with shadow
- **Chat Page**: Light blue gradient background (#dbeafe)
- **Catalog/Data Pages**: White content area with shadow
- **Primary Action**: Light blue (#dbeafe) → Blue (#3b82f6) on hover
- **Text**: Dark gray (#181C3A)

### Layout Pattern
All pages follow the same structure:
1. **Fixed Global Header** (with navigation)
2. **Content Area** (page-specific layout)
3. **No footer on Chat/Catalog/Data** (maximized vertical space)

### Typography
- **Font Family**: Segoe UI (system font)
- **Headers**: Bold, dark navy
- **Body Text**: Regular weight
- **Tables**: Slightly smaller (0.97rem) for data density

## 🔄 User Journey Flows

### Typical Workflow 1: Chat about data
```
Home → Click "Chat" → Click "New Chat" → Type question → View AI response → Click suggested prompt → Continue conversation
```

### Typical Workflow 2: Browse and explore data
```
Home → Click "Catalog" → Browse datasets → Click interesting dataset → View data table → Apply filters → Analyze results
```

### Typical Workflow 3: Direct data lookup
```
Home → Click "Data" → Type table code → View table → Apply column filters → Sort by value → Analyze filtered data
```

---

**Version 4 Approach**: UI-based architecture showing what users actually see and interact with, organized by visual screens rather than technical components. Focuses on layout, visual hierarchy, and user interaction patterns.

