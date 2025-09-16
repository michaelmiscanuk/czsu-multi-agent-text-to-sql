# API Server Refactoring Checklist

## Overview
This checklist guides the step-by-step refactoring of `api_server.py` (3210 lines) into a modular structure.

**⚠️ CRITICAL**: Always test after each major step to ensure nothing is broken.

**🔧 CRITICAL INITIALIZATION CODE**: The following code block MUST be duplicated at the very beginning of each script that uses asyncio or database connections, BEFORE any other imports:

```python
# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Only after this can other LOCAL imports happen, like:
from my_agent.utils.postgres_checkpointer import (
    check_postgres_env_vars, 
    get_db_config
)
```

**📝 TESTING STRATEGY**: Each phase includes a specific test file that validates the current step's functionality using the pattern from `test_concurrency.py` - here we always need to use functionality from the main script as possible, importing the functions from there, not implementing it again!! You can use same satup for google jwt test tokens if needed.

**🎯 REFACTORING APPROACH**: 
1. **Extract complete sections** from `api_server.py` (marked by `#============================================================`)
2. **Add necessary imports and critical initialization code** to each new file
3. **Update external files** that import from `api_server.py` to use new modular structure
4. **Create test files** for each extracted section
5. **Replace api_server.py** only at the very end with a simple import wrapper

**📝 DO NOT MODIFY api_server.py** during the refactoring process - only extract from it!

**🔗 FRONTEND COMPATIBILITY**: Each route extraction phase includes checking frontend usage and updating paths to ensure the frontend continues to work correctly with the new modular API structure.

---

## Phase 1: Setup and Preparation

### 1.1 Create Folder Structure
- [ ] Create `api/` folder in project root
- [ ] Create `api/__init__.py`
- [ ] Create `api/config/` folder
- [ ] Create `api/config/__init__.py`
- [ ] Create `api/utils/` folder
- [ ] Create `api/utils/__init__.py`
- [ ] Create `api/models/` folder
- [ ] Create `api/models/__init__.py`
- [ ] Create `api/middleware/` folder
- [ ] Create `api/middleware/__init__.py`
- [ ] Create `api/auth/` folder
- [ ] Create `api/auth/__init__.py`
- [ ] Create `api/exceptions/` folder
- [ ] Create `api/exceptions/__init__.py`
- [ ] Create `api/dependencies/` folder
- [ ] Create `api/dependencies/__init__.py`
- [ ] Create `api/routes/` folder
- [ ] Create `api/routes/__init__.py`
- [ ] Create `tests/` folder for refactoring tests
- [ ] Create `tests/__init__.py`

### 1.2 Create Base Test Template
- [ ] Create `tests/test_phase1_setup.py` (validates folder structure and imports)
- [ ] Test: Run `python tests/test_phase1_setup.py` to verify setup 

---

## Phase 2: Extract Configuration and Constants

### 2.1 Extract Configuration Section
- [ ] Create `api/config/settings.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete "CONFIGURATION AND CONSTANTS" section** (lines ~127-178):
  - [ ] Copy the entire section including comments
  - [ ] Add necessary imports (`import time`, `import asyncio`, `from collections import defaultdict`)
  - [ ] Include all constants:
    - [ ] `start_time`
    - [ ] `GC_MEMORY_THRESHOLD`
    - [ ] `_app_startup_time`, `_memory_baseline`, `_request_count`
    - [ ] `GLOBAL_CHECKPOINTER`
    - [ ] `MAX_CONCURRENT_ANALYSES`, `analysis_semaphore`
    - [ ] All rate limiting constants and variables
    - [ ] `BULK_CACHE_TIMEOUT`, `_bulk_loading_cache`, `_bulk_loading_locks`
    - [ ] `GOOGLE_JWK_URL`, `_jwt_kid_missing_count`
- [ ] **Extract environment variable reading** (lines ~88-89, 141-142, 155-156):
  - [ ] `INMEMORY_FALLBACK_ENABLED`
  - [ ] `BASE_DIR` calculation
- [ ] Create `tests/test_phase2_config.py` (validates configuration loading by importing from `api.config.settings`)
- [ ] Test: Run `python tests/test_phase2_config.py` to ensure configuration is loaded correctly

---

## Phase 3: Extract Utility Functions

### 3.1 Extract Debug Utilities Section
- [ ] Create `api/utils/debug.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete debug functions section** (lines ~518-778):
  - [ ] Copy entire section with all debug print functions
  - [ ] Add necessary imports (`import os`, `import sys`)
  - [ ] Include all debug functions from `print__api_postgresql` to `print__analysis_tracing_debug`
  - [ ] Keep `print__startup_debug` in main for now (will move later)
- [ ] Create `tests/test_phase3_debug.py` (validates debug functions by importing and testing each one)
- [ ] Test: Run `python tests/test_phase3_debug.py` to verify debug functions work correctly

### 3.2 Extract Memory Management Section
- [ ] Create `api/utils/memory.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete "UTILITY FUNCTIONS - MEMORY MANAGEMENT" section** (lines ~179-399):
  - [ ] Copy entire section including `cleanup_bulk_cache`, `check_memory_and_gc`, etc.
  - [ ] Add necessary imports (`import gc`, `import psutil`, `import signal`, `import time`, etc.)
  - [ ] Move `print__memory_monitoring` from line ~115 to this file
  - [ ] Include `setup_graceful_shutdown` and `perform_deletion_operations`
- [ ] Import global variables from `api.config.settings` that are needed
- [ ] Create `tests/test_phase3_memory.py` (validates memory functions by importing from `api.utils.memory`)
- [ ] Test: Run `python tests/test_phase3_memory.py` to verify memory monitoring works correctly

### 3.3 Extract Rate Limiting Utilities
- [ ] Create `api/utils/rate_limiting.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract rate limiting functions** (lines ~285-380):
  - [ ] Copy `check_rate_limit_with_throttling`, `wait_for_rate_limit`, `check_rate_limit`
  - [ ] Add necessary imports (`import time`, `import asyncio`, `from collections import defaultdict`)
  - [ ] Import rate limiting globals from `api.config.settings`
- [ ] Create `tests/test_phase3_rate_limiting.py` (validates rate limiting by importing from new module)
- [ ] Test: Run `python tests/test_phase3_rate_limiting.py` to verify rate limiting works correctly

---

## Phase 4: Extract Models

### 4.1 Extract Pydantic Models Section
- [ ] Create `api/models/requests.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete "PYDANTIC MODELS" section request models** (lines ~981-1023):
  - [ ] Copy `AnalyzeRequest`, `FeedbackRequest`, `SentimentRequest` with all validators
  - [ ] Add necessary imports (`from pydantic import BaseModel, Field, field_validator`, `from typing import Optional`, `import uuid`)
- [ ] Create `api/models/responses.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract response models** (lines ~1041-1070):
  - [ ] Copy `ChatThreadResponse`, `PaginatedChatThreadsResponse`, `ChatMessage`
  - [ ] Add necessary imports (`from datetime import datetime`, `from typing import List, Optional`)
- [ ] Create `tests/test_phase4_models.py` (validates models by importing and testing validation)
- [ ] Test: Run `python tests/test_phase4_models.py` to verify models work correctly

---

## Phase 5: Extract Authentication

### 5.1 Extract Authentication Section
- [ ] Create `api/auth/jwt_auth.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete "AUTHENTICATION" section** (lines ~1073-1235):
  - [ ] Copy entire `verify_google_jwt` function with all its logic
  - [ ] Add necessary imports (`import jwt`, `import requests`, `import time`, `from jwt.algorithms import RSAAlgorithm`, etc.)
  - [ ] Import constants from `api.config.settings` (`GOOGLE_JWK_URL`, `_jwt_kid_missing_count`)
- [ ] Create `api/dependencies/auth.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract authentication dependencies** (lines ~1236-1318):
  - [ ] Copy `get_current_user` function
  - [ ] Add necessary imports (`from fastapi import Header, HTTPException`, etc.)
  - [ ] Import `verify_google_jwt` from `api.auth.jwt_auth`
- [ ] Create `tests/test_phase5_auth.py` (validates authentication by importing and testing with test tokens)
- [ ] Test: Run `python tests/test_phase5_auth.py` to verify authentication works correctly

---

## Phase 6: Extract Exception Handlers

### 6.1 Extract Exception Handlers Section
- [ ] Create `api/exceptions/handlers.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete exception handlers** (lines ~919-977):
  - [ ] Copy all exception handler functions
  - [ ] Add necessary imports (`from fastapi import Request`, `from fastapi.responses import JSONResponse`, etc.)
  - [ ] Import debug functions from `api.utils.debug`
- [ ] Create `tests/test_phase6_exceptions.py` (validates exception handlers)
- [ ] Test: Run `python tests/test_phase6_exceptions.py` to verify exception handling works correctly

---

## Phase 7: Extract Middleware

### 7.1 Extract Middleware Sections
- [ ] Create `api/middleware/cors.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract CORS middleware setup** (lines ~837-845)
- [ ] Create `api/middleware/rate_limiting.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract throttling middleware** (lines ~860-894):
  - [ ] Copy complete middleware function
  - [ ] Import rate limiting functions from `api.utils.rate_limiting`
  - [ ] Import helper functions from appropriate modules
- [ ] Create `api/middleware/memory_monitoring.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract memory monitoring middleware** (lines ~895-918):
  - [ ] Copy complete middleware function
  - [ ] Import memory functions from `api.utils.memory`
- [ ] Create `tests/test_phase7_middleware.py` (validates middleware setup)
- [ ] Test: Run `python tests/test_phase7_middleware.py` to verify middleware works correctly

---

## Phase 8: Extract Routes (Order matters - start with simplest)

### 8.1 Extract Health Check Routes
- [ ] Create `api/routes/health.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract complete health endpoints section** (lines ~1319-1580):
  - [ ] Copy all health check functions with their decorators
  - [ ] Add necessary imports (`from fastapi import APIRouter`, `from datetime import datetime`, etc.)
  - [ ] Import dependencies from appropriate modules
  - [ ] Create router: `router = APIRouter()`
  - [ ] Replace `@app.get` with `@router.get`
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for health check endpoint calls (e.g., `/health`, `/health/database`)
  - [ ] Update any hardcoded paths if needed
  - [ ] Verify API documentation matches new structure
- [ ] Create `tests/test_phase8_health.py` (validates health endpoints by making HTTP requests)
- [ ] Test: Run `python tests/test_phase8_health.py` to verify health endpoints work correctly

### 8.2 Extract Catalog Routes
- [ ] Create `api/routes/catalog.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract catalog endpoints** (lines ~2215-2299):
  - [ ] Copy `get_catalog`, `get_data_tables`, `get_data_table` functions
  - [ ] Add necessary imports and create router
  - [ ] Import authentication dependencies
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for catalog endpoint calls (e.g., `/catalog`, `/data-tables`)
  - [ ] Update any hardcoded paths in frontend components
  - [ ] Verify TypeScript interfaces match response models
- [ ] Create `tests/test_phase8_catalog.py` (validates catalog endpoints)
- [ ] Test: Run `python tests/test_phase8_catalog.py` to verify catalog endpoints work correctly

### 8.3 Extract Analysis Routes
- [x] Create `api/routes/analysis.py` with CRITICAL INITIALIZATION CODE at top
- [x] **Extract analysis endpoint** (lines ~1581-1764):
  - [x] Copy complete `analyze` function
  - [x] Import all necessary modules (`from main import main as analysis_main`, etc.)
  - [x] Import models from `api.models.requests`
  - [x] Import authentication from `api.dependencies.auth`
  - [x] Import globals from `api.config.settings`
- [x] **Check frontend usage and update paths**:
  - [x] Search frontend code for analysis endpoint calls (e.g., `/analyze`)
  - [x] Update any hardcoded paths in frontend components
  - [x] Verify request/response models match frontend TypeScript interfaces
  - [x] Check if frontend uses any specific headers or authentication patterns
- [x] Create `tests/test_phase8_analysis.py` (validates analysis endpoint - based on test_concurrency.py)
- [x] Test: Run `python tests/test_phase8_analysis.py` to verify analysis endpoint works correctly

### 8.4 Extract Feedback Routes
- [ ] Create `api/routes/feedback.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract feedback endpoints** (lines ~1765-1995):
  - [ ] Copy `submit_feedback` and `update_sentiment` functions
  - [ ] Add necessary imports and authentication
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for feedback endpoint calls (e.g., `/feedback`, `/sentiment`)
  - [ ] Update any hardcoded paths in frontend components
  - [ ] Verify feedback forms and handlers use correct endpoints
- [ ] Create `tests/test_phase8_feedback.py` (validates feedback endpoints)
- [ ] Test: Run `python tests/test_phase8_feedback.py` to verify feedback endpoints work correctly

### 8.5 Extract Chat Routes
- [ ] Create `api/routes/chat.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract chat management endpoints** (lines ~1996-2214):
  - [ ] Copy `get_thread_sentiments`, `get_chat_threads`, `delete_chat_checkpoints`
  - [ ] Import helper functions from `api.utils.memory` and other modules
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for chat endpoint calls (e.g., `/chat/threads`, `/chat/sentiments`)
  - [ ] Update any hardcoded paths in frontend components
  - [ ] Verify chat components use correct endpoints
  - [ ] Check pagination and filtering parameters
- [ ] Create `tests/test_phase8_chat.py` (validates chat endpoints)
- [ ] Test: Run `python tests/test_phase8_chat.py` to verify chat endpoints work correctly

### 8.6 Extract Message Routes
- [ ] Create `api/routes/messages.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract message endpoints** (lines ~2300-2525):
  - [ ] Copy `get_chat_messages` and `get_message_run_ids` functions
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for message endpoint calls (e.g., `/messages`, `/chat/{thread_id}/messages`)
  - [ ] Update any hardcoded paths in frontend components
  - [ ] Verify message display components use correct endpoints
  - [ ] Check real-time message updates if using WebSockets
- [ ] Create `tests/test_phase8_messages.py` (validates message endpoints)
- [ ] Test: Run `python tests/test_phase8_messages.py` to verify message endpoints work correctly

### 8.7 Extract Bulk Operations Routes
- [ ] Create `api/routes/bulk.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract bulk operations** (lines ~2526-2891):
  - [ ] Copy complete `get_all_chat_messages` function including nested functions
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for bulk operation endpoint calls (e.g., `/bulk/messages`)
  - [ ] Update any hardcoded paths in frontend components
  - [ ] Verify bulk export/import features use correct endpoints
  - [ ] Check if frontend has loading states for bulk operations
- [ ] Create `tests/test_phase8_bulk.py` (validates bulk operations)
- [ ] Test: Run `python tests/test_phase8_bulk.py` to verify bulk operations work correctly

### 8.8 Extract Debug Routes
- [ ] Create `api/routes/debug.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract debug endpoints** (lines ~2892-3150):
  - [ ] Copy all debug and admin endpoints
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for debug endpoint calls (e.g., `/debug/pool`, `/admin/cache`)
  - [ ] Update any hardcoded paths in frontend admin components
  - [ ] Verify admin panels use correct endpoints
  - [ ] Check if debug endpoints are properly protected in frontend
- [ ] Create `tests/test_phase8_debug.py` (validates debug endpoints)
- [ ] Test: Run `python tests/test_phase8_debug.py` to verify debug endpoints work correctly

### 8.9 Extract Miscellaneous Routes
- [ ] Create `api/routes/misc.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract miscellaneous endpoints** (lines ~3151-3210):
  - [ ] Copy `get_placeholder_image`
- [ ] **Check frontend usage and update paths**:
  - [ ] Search frontend code for miscellaneous endpoint calls (e.g., `/placeholder-image`)
  - [ ] Update any hardcoded paths in frontend components
  - [ ] Verify image loading and other misc features use correct endpoints
- [ ] Create `tests/test_phase8_misc.py` (validates miscellaneous endpoints)
- [ ] Test: Run `python tests/test_phase8_misc.py` to verify miscellaneous endpoints work correctly

---

## Phase 9: Create Main Application File

### 9.1 Create Main FastAPI Application
- [ ] Create `api/main.py` with CRITICAL INITIALIZATION CODE at top
- [ ] **Extract application setup sections**:
  - [ ] Copy lifespan management (lines ~781-836)
  - [ ] Copy FastAPI app creation (lines ~837-859)
  - [ ] Import all middleware from `api.middleware.*`
  - [ ] Import all exception handlers from `api.exceptions.handlers`
  - [ ] Import all route routers and include them: `app.include_router(health_router)`
- [ ] Create `tests/test_phase9_main.py` (validates main application assembly)
- [ ] Test: Run `python tests/test_phase9_main.py` to verify main application works correctly

---

## Phase 10: Update External File Imports

### 10.1 Update Files That Import api_server
- [ ] **Find all files that import from api_server**:
  - [ ] Search for `from api_server import` in all project files
  - [ ] Search for `import api_server` in all project files
  - [ ] Common files to check: `main.py`, `uvicorn_start.py`, test files
- [ ] **Update import statements**:
  - [ ] Replace `from api_server import app` with `from api.main import app`
  - [ ] Replace other specific imports with new module paths
- [ ] Create `tests/test_phase10_external_imports.py` (validates external file imports work)
- [ ] Test: Run `python tests/test_phase10_external_imports.py` to verify external imports work

---

## Phase 11: Replace Original api_server.py

### 11.1 Create Import Wrapper
- [ ] **Rename original file**: `mv api_server.py api_server_original.py`
- [ ] **Create new api_server.py** with simple import wrapper:
```python
# CRITICAL: Set Windows event loop policy FIRST
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Backward compatibility wrapper
from api.main import app

# Export commonly used items for backward compatibility
from api.dependencies.auth import get_current_user
from api.models.requests import AnalyzeRequest, FeedbackRequest, SentimentRequest
from api.models.responses import ChatMessage, ChatThreadResponse

__all__ = ['app', 'get_current_user', 'AnalyzeRequest', 'FeedbackRequest', 'SentimentRequest', 'ChatMessage', 'ChatThreadResponse']
```
- [ ] Create `tests/test_phase11_wrapper.py` (validates backward compatibility)
- [ ] Test: Run `python tests/test_phase11_wrapper.py` to verify wrapper works correctly

---

## Phase 12: Final Testing and Cleanup

### 12.1 Comprehensive Testing
- [ ] Create `tests/test_phase12_comprehensive.py` (runs all endpoint tests using new structure)
- [ ] Test all endpoints work through new modular structure
- [ ] Test authentication flow
- [ ] Test error handling
- [ ] Test memory monitoring
- [ ] Test rate limiting
- [ ] Test: Run `python tests/test_phase12_comprehensive.py` to run comprehensive tests

### 12.2 Performance Testing
- [ ] Create `tests/test_phase12_performance.py` (based on test_concurrency.py pattern)
- [ ] Test application startup time with new structure
- [ ] Test memory usage patterns
- [ ] Test concurrent request handling
- [ ] Test: Run `python tests/test_phase12_performance.py` to verify performance is maintained

### 12.3 Final Validation
- [ ] Create `tests/test_phase12_final.py` (end-to-end validation)
- [ ] Full end-to-end testing
- [ ] Integration testing with frontend
- [ ] Database connectivity testing
- [ ] Authentication testing with real tokens
- [ ] Test: Run `python tests/test_phase12_final.py` to run final validation

### 12.4 Cleanup
- [ ] Remove `api_server_original.py` after confirming everything works
- [ ] Update documentation with new structure
- [ ] Update type hints if needed
- [ ] Run linting tools on new modules

---

## Test File Templates

### Basic Test Structure (Based on test_concurrency.py)
All test files should follow this pattern:

```python
"""
Test for Phase X: [Description]
Based on test_concurrency.py pattern - imports functionality from main scripts
"""

# CRITICAL: Set Windows event loop policy FIRST, before other imports
import sys
import os
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

# Constants
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import asyncio
import time
import httpx
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import functionality from main scripts (not reimplementing!)
from other.tests.test_concurrency import create_test_jwt_token, check_server_connectivity

# Test configuration
SERVER_BASE_URL = os.environ.get("TEST_SERVER_URL")
REQUEST_TIMEOUT = 30.0

async def test_phase_x_functionality():
    """Test specific phase functionality by importing from new modules."""
    # Import from the new modular structure and test it
    # Example: from api.config.settings import GC_MEMORY_THRESHOLD
    pass

async def main():
    """Main test runner."""
    print(f"Testing Phase X functionality...")
    
    # Check server connectivity using existing function
    if not await check_server_connectivity():
        return False
    
    # Run phase-specific tests
    await test_phase_x_functionality()
    
    print("✅ Phase X tests completed successfully")
    return True

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Notes

**Critical Points to Remember:**
1. **DO NOT modify api_server.py** during refactoring - only extract from it
2. Extract **complete sections** with their comment headers
3. Always add **CRITICAL INITIALIZATION CODE** at the top of every new file
4. **Import existing functionality** in tests - don't reimplement
5. Update **external files** that import from api_server, not api_server itself
6. Test after each major phase to catch issues early
7. Replace api_server.py **only at the very end** with import wrapper

**Files that will be created:**
- 25+ new modular files
- 25+ test files covering each phase
- All based on extracting complete sections

**Files that will be modified:**
- External files that import from `api_server`
- `api_server.py` (only at the very end - becomes import wrapper)

**Estimated total time:** 8-12 hours for careful, tested refactoring with comprehensive testing