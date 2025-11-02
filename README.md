# CZSU Multi-Agent Text-to-SQL Application (CZSU Data Explorer and Chatbot)

This is a full-stack application prototype that enables natural language querying of Czech Statistics Office (CZSU) data through an AI-powered chatbot. The system features a Python/FastAPI backend, React frontend, and an agentic flow using LangGraph for intelligent question answering over statistical data. The AI component is evaluated using LangSmith.

The application allows users to ask questions in natural language about CZSU data selections, which include various metrics and dimensions. It includes comprehensive data and metadata preparation scripts, backend API endpoints, and a web interface for interacting with the chatbot.

# Setup Instructions

## Windows

### Option 1: Automated Setup
- To set up all backend and frontend dependencies, run `setup.bat` from the root folder.
-- If you already have dependencies install, and want to do clean setup - remove there items first:
--- in the root: .venv folder; in frontend folder: node_modules folder, .next folder and package-lock file
- To start the backend, run `start_backend.bat` from the root folder.
- To start the frontend, run `start_frontend.bat` from the `frontend` folder.

### Option 2: Manual Setup
1. **Environment Variables**
   - Create a `.env` file in the root directory based on `.env_example`.
   - Create a `.env` file in the `frontend` folder based on `.env_example`.
2. **Backend Setup (from root folder)**
   Run the following commands in CMD:
   ```
   uv venv --python 3.11.9
   .venv\Scripts\activate
   uv pip install .
   uv pip install .[dev]
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info --reload
   ```
   
   If there is some problem with libraries, delete folders:  
   (.venv, czsu_multi_agent_text_to_sql.egg-info, __pycache__)
   Run commands above.
   
3. **Frontend Setup (from frontend folder)**
   Run the following commands:
   ```
   npm install --verbose
   npm run build
   npm run dev
   ```

   If there isome problem with libraries, delete these items in frontend folder:
   (node_modules folder, .next folder and package-lock file)
   Run commands above.
   
   If more problems, try again deleting same folders as above and try with China Registry:
   
   ```
   npm set registry https://registry.npmmirror.com/
   npm install --verbose
   ```

---

## Deployment

For deployment to platforms like Railway, the `railway.toml` file contains the necessary configuration for setting up the application in a Linux environment. Ensure your environment variables are configured as per the `.env_example` files, and the deployment service will handle the build and startup processes automatically.
---

## Running Notebooks
You can run notebooks in VS Code directly:

1. Install the VS Code Python extension if not already installed
2. Open any .ipynb file in VS Code
3. Select the `.venv` environment as the kernel in the Python interpreter selection
4. Run cells using the play button or Shift+Enter

---

## MCP Server

This project includes a standalone MCP (Model Context Protocol) server for SQLite queries.

### Default: Local SQLite Mode

By default, the application uses local SQLite - no MCP server needed.

**`.env` configuration:**
```env
MCP_SERVER_URL=
USE_LOCAL_SQLITE_FALLBACK=1
```

### Optional: Remote MCP Server

Deploy the MCP server to FastMCP Cloud for centralized database access.

**Quick setup:**

1. Deploy MCP server (see `czsu_mcp_server_sqlite/README.md`)
2. Update `.env`:
   ```env
   MCP_SERVER_URL=https://your-project-name.fastmcp.app/mcp
   USE_LOCAL_SQLITE_FALLBACK=1
   ```

### Documentation

- **Quick Start**: `MCP_QUICKSTART.md`
- **Implementation**: `MCP_IMPLEMENTATION_SUMMARY.md`
- **MCP Server**: `czsu_mcp_server_sqlite/README.md`
- **Deployment**: `czsu_mcp_server_sqlite/docs/DEPLOYMENT.md`

