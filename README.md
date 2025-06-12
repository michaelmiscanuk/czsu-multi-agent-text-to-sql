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
   uvicorn api_server:app --reload --reload-exclude .venv
   ```
   
   If there is some problem with libraries, delete .venv folder and run commands above
   
3. **Frontend Setup (from frontend folder)**
   Run the following commands:
   ```
   npm install
   npm run build
   npm run dev
   ```

   If there isome problem with libraries, delete these items in frontend folder
   node_modules folder, .next folder and package-lock file
   Run commands above.

---

## Unix/Linux/macOS

### Option 1: Automated Setup
- To set up all backend and frontend dependencies, run `./setup.sh` from the root folder.
- To start the backend, run `./start_backend.sh` from the root folder.
- To start the frontend, run `./start_frontend.sh` from the `frontend` folder.

### Option 2: Manual Setup
1. **Environment Variables**
   - Create a `.env` file in the root directory based on `.env_example`.
   - Create a `.env` file in the `frontend` folder based on `.env_example`.
2. **Backend Setup (from root folder)**
   Run the following commands in your shell:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install .
   pip install .[dev]
   uvicorn api_server:app --reload --reload-exclude .venv
   ```
   
   If there is some problem with libraries, delete .venv folder and run commands above
   
3. **Frontend Setup (from frontend folder)**
   Run the following commands:
   ```
   npm install
   npm run build
   npm run dev
   ```

   If there isome problem with libraries, delete these items in frontend folder
   node_modules folder, .next folder and package-lock file
   Run commands above.
---

## Running Notebooks
You can run notebooks in VS Code directly:

1. Install the VS Code Python extension if not already installed
2. Open any .ipynb file in VS Code
3. Select the `.venv` environment as the kernel in the Python interpreter selection
4. Run cells using the play button or Shift+Enter
