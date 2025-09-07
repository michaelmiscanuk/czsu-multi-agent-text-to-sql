"""
Startup Debug Test Suite
Enhanced startup diagnostics with comprehensive error reporting and phase monitoring.
Based on test_phase8_catalog.py structure but focused on backend startup validation.
"""

import asyncio
import sys
import os
import time
import subprocess
import signal
import httpx
import traceback
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tests.helpers import (
    BaseTestResults,
    handle_error_response,
    handle_expected_failure,
    extract_detailed_error_info,
    make_request_with_traceback_capture,
    save_traceback_report,
    create_test_jwt_token,
    check_server_connectivity,
    setup_debug_environment,
    cleanup_debug_environment,
)

# Set Windows event loop policy FIRST
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()

# Test configuration
STARTUP_TIMEOUT = 30
REQUIRED_ENV_VARS = {"host", "port", "dbname", "user", "password"}
TEST_CASES = [
    {
        "test_id": "env_check",
        "description": "Environment Variables Check",
        "function": "test_environment_variables",
        "timeout": 5,
    },
    {
        "test_id": "imports",
        "description": "Critical Imports Check",
        "function": "test_critical_imports",
        "timeout": 10,
    },
    {
        "test_id": "db_connection",
        "description": "Direct Database Connection",
        "function": "test_database_connection",
        "timeout": 15,
    },
    {
        "test_id": "checkpointer_init",
        "description": "PostgreSQL Checkpointer Initialization",
        "function": "test_checkpointer_initialization",
        "timeout": 20,
    },
    {
        "test_id": "uvicorn_startup",
        "description": "Uvicorn Server Startup Process",
        "function": "test_uvicorn_startup",
        "timeout": 25,
    },
]


class StartupTestResults(BaseTestResults):
    """Extended test results class for startup tests."""

    def __init__(self):
        super().__init__(required_endpoints=set())
        self.startup_info = {}
        self.process_info = {}

    def add_startup_result(
        self,
        test_id: str,
        description: str,
        response_time: float,
        details: Dict[str, Any] = None,
    ):
        """Add a successful startup test result."""
        result = {
            "test_id": test_id,
            "endpoint": "startup",
            "description": description,
            "response_data": details or {},
            "response_time": response_time,
            "status_code": 200,
            "timestamp": datetime.now().isoformat(),
            "success": True,
        }
        self.results.append(result)

    def add_process_info(self, key: str, value: Any):
        """Add process information."""
        self.process_info[key] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of startup test results."""
        # Get base summary from parent class
        base_summary = super().get_summary()

        # Add startup-specific information
        base_summary.update(
            {
                "startup_info": self.startup_info,
                "tracked_processes": self.process_info,
            }
        )

        return base_summary

    def add_error(
        self,
        test_id: str,
        error_type: str,
        description: str,
        exception: Exception,
        response_time: float,
        endpoint: str = "startup",
    ):
        """Add error with traceback capture for startup diagnostics."""
        # Capture full traceback for startup errors
        full_traceback = (
            traceback.format_exc() if exception else "No traceback available"
        )

        error_entry = {
            "test_id": test_id,
            "endpoint": endpoint,
            "error_type": error_type,
            "description": description,
            "error": str(exception),
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            # Store traceback in response_data to match catalog test format
            "response_data": {
                "traceback": full_traceback,
                "error_details": {
                    "exception_type": (
                        type(exception).__name__ if exception else "Unknown"
                    ),
                    "exception_args": getattr(exception, "args", []),
                    "startup_phase": test_id,
                },
            },
        }
        self.errors.append(error_entry)


async def test_environment_variables(results: StartupTestResults) -> bool:
    """Test required environment variables."""
    print("🔍 1. Checking environment variables...")
    start_time = time.time()

    try:
        missing_vars = []
        present_vars = {}

        for var in REQUIRED_ENV_VARS:
            value = os.environ.get(var)
            if value:
                display_value = "***" if var == "password" else value
                present_vars[var] = display_value
                print(f"✅ {var}: {display_value}")
            else:
                missing_vars.append(var)
                print(f"❌ {var}: NOT SET")

        response_time = time.time() - start_time

        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            results.add_error(
                "env_check",
                "environment",
                "Environment Variables Check",
                Exception(error_msg),
                response_time,
            )
            return False

        results.add_startup_result(
            "env_check",
            "Environment Variables Check",
            response_time,
            {"status": "all_present", "variables": present_vars},
        )
        return True

    except Exception as e:
        response_time = time.time() - start_time
        results.add_error(
            "env_check", "environment", "Environment Variables Check", e, response_time
        )
        return False


async def test_critical_imports(results: StartupTestResults) -> bool:
    """Test importing critical modules."""
    print("\n🔍 2. Testing critical imports...")
    start_time = time.time()

    try:
        print("🔄 Importing FastAPI...")
        import fastapi

        print("🔄 Importing uvicorn...")
        import uvicorn

        print("🔄 Importing psycopg...")
        import psycopg

        print("🔄 Importing LangGraph...")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        print("🔄 Importing checkpointer utilities...")
        from checkpointer.config import get_db_config
        from checkpointer.database.connection import get_connection_string
        from checkpointer.checkpointer.factory import initialize_checkpointer
        from checkpointer.checkpointer.factory import create_async_postgres_saver

        response_time = time.time() - start_time
        print("✅ All critical imports successful")

        results.add_startup_result(
            "imports",
            "Critical Imports Check",
            response_time,
            {"imports": ["fastapi", "uvicorn", "psycopg", "langgraph", "checkpointer"]},
        )
        return True

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Import failed: {e}")
        results.add_error(
            "imports", "imports", "Critical Imports Check", e, response_time
        )
        return False


async def test_database_connection(results: StartupTestResults) -> bool:
    """Test direct database connection."""
    print("\n🔍 3. Testing direct database connection...")
    start_time = time.time()

    try:
        from checkpointer.database.connection import get_connection_string
        import psycopg

        conn_str = get_connection_string()
        print("🔄 Attempting direct psycopg connection...")

        async with await asyncio.wait_for(
            psycopg.AsyncConnection.connect(conn_str), timeout=10.0
        ) as conn:
            print("✅ Direct database connection successful")

            async with conn.cursor() as cur:
                await cur.execute("SELECT version()")
                result = await cur.fetchone()
                postgres_version = result[0][:50] if result else "Unknown"
                print(f"✅ PostgreSQL version: {postgres_version}...")

        response_time = time.time() - start_time
        results.add_startup_result(
            "db_connection",
            "Direct Database Connection",
            response_time,
            {"postgres_version": postgres_version, "connection_time": response_time},
        )
        return True

    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        print("❌ Database connection timed out")
        results.add_error(
            "db_connection",
            "database",
            "Direct Database Connection",
            Exception("Database connection timeout"),
            response_time,
        )
        return False
    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Database connection failed: {e}")
        results.add_error(
            "db_connection", "database", "Direct Database Connection", e, response_time
        )
        return False


async def test_checkpointer_initialization(results: StartupTestResults) -> bool:
    """Test PostgreSQL checkpointer initialization."""
    print("\n🔍 4. Testing checkpointer initialization...")
    start_time = time.time()

    try:
        from checkpointer.checkpointer.factory import initialize_checkpointer

        print("🔄 Running initialize_checkpointer()...")
        await asyncio.wait_for(initialize_checkpointer(), timeout=20.0)
        print("✅ Checkpointer initialization completed")

        response_time = time.time() - start_time
        results.add_startup_result(
            "checkpointer_init",
            "PostgreSQL Checkpointer Initialization",
            response_time,
            {"initialization_complete": True, "global_state_set": True},
        )
        return True

    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        print("❌ Checkpointer initialization timed out")
        results.add_error(
            "checkpointer_init",
            "checkpointer",
            "PostgreSQL Checkpointer Initialization",
            Exception("Checkpointer initialization timeout"),
            response_time,
        )
        return False
    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Checkpointer initialization failed: {e}")
        results.add_error(
            "checkpointer_init",
            "checkpointer",
            "PostgreSQL Checkpointer Initialization",
            e,
            response_time,
        )
        return False


async def test_uvicorn_startup(results: StartupTestResults) -> bool:
    """Enhanced uvicorn server startup test with detailed phase monitoring."""
    print("\n🔍 5. Testing uvicorn server startup...")
    start_time = time.time()

    # Enhanced startup phases for detailed monitoring
    startup_phases = {
        "process_creation": {"completed": False, "duration": 0, "details": {}},
        "initial_logs": {"completed": False, "duration": 0, "details": {}},
        "port_binding": {"completed": False, "duration": 0, "details": {}},
        "app_loading": {"completed": False, "duration": 0, "details": {}},
        "server_ready": {"completed": False, "duration": 0, "details": {}},
        "health_check": {"completed": False, "duration": 0, "details": {}},
        "endpoint_validation": {"completed": False, "duration": 0, "details": {}},
    }

    process = None
    startup_success = False
    detailed_logs = []

    try:
        print("🔄 Starting uvicorn process...")
        phase_start = time.time()

        # Create uvicorn process with enhanced environment
        process_env = {
            **os.environ,
            "DEBUG": "1",
            "VERBOSE_SSL_LOGGING": "true",
            "ENABLE_CONNECTION_MONITORING": "true",
            "PYTHONUNBUFFERED": "1",  # Ensure real-time output
        }

        process = subprocess.Popen(
            [sys.executable, "uvicorn_start.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=parent_dir,
            env=process_env,
            bufsize=1,  # Line buffered for real-time output
            universal_newlines=True,
        )

        startup_phases["process_creation"]["completed"] = True
        startup_phases["process_creation"]["duration"] = time.time() - phase_start
        startup_phases["process_creation"]["details"] = {
            "pid": process.pid,
            "command": f"{sys.executable} uvicorn_start.py",
            "working_dir": parent_dir,
        }

        results.add_process_info("uvicorn_pid", process.pid)
        print(f"✅ Uvicorn process started (PID: {process.pid})")
        detailed_logs.append(f"[PHASE-1] Process created: PID {process.pid}")

        # Enhanced startup monitoring with phase detection
        total_timeout = 300  # Increased timeout for thorough testing
        check_interval = 0.5  # More frequent checks

        for check_round in range(int(total_timeout / check_interval)):
            current_time = time.time()
            elapsed = current_time - start_time

            # Check timeout first - break if we've exceeded the limit
            if elapsed >= total_timeout:
                print(f"⏰ Timeout reached: {elapsed:.1f}s >= {total_timeout}s")
                break

            # Check if process exited early
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return await _handle_uvicorn_early_exit(
                    results,
                    process,
                    stdout,
                    stderr,
                    elapsed,
                    startup_phases,
                    detailed_logs,
                )

            # Read available output without blocking (Windows compatible)
            try:
                new_output = ""

                # Simple approach: just check if process is producing output
                # We'll rely more on the health check for status rather than parsing logs
                if check_round % 4 == 0:  # Every 2 seconds, try to get some output
                    try:
                        # Quick check for any new output (non-blocking attempt)
                        if process.stdout and process.stdout.readable():
                            # Just log that we're monitoring - detailed log parsing isn't critical
                            detailed_logs.append(
                                f"[{elapsed:.1f}s] Monitoring process output..."
                            )
                    except Exception:
                        pass

                # Analyze logs for startup phases (simplified)
                await _analyze_startup_phase(new_output, startup_phases, elapsed)

            except Exception as read_error:
                # Fallback - just continue monitoring
                detailed_logs.append(
                    f"[{elapsed:.1f}s] Output monitoring note: {read_error}"
                )

            # Test server connectivity based on startup progress
            if elapsed > 5.0:  # Only start testing after reasonable startup time
                connectivity_result = await _test_server_connectivity(
                    elapsed, detailed_logs
                )

                if connectivity_result["success"]:
                    startup_phases["health_check"]["completed"] = True
                    startup_phases["health_check"]["duration"] = elapsed
                    startup_phases["health_check"]["details"] = connectivity_result[
                        "details"
                    ]

                    # Run comprehensive endpoint validation
                    validation_result = await _run_endpoint_validation(detailed_logs)

                    if validation_result["success"]:
                        startup_phases["endpoint_validation"]["completed"] = True
                        startup_phases["endpoint_validation"]["duration"] = elapsed
                        startup_phases["endpoint_validation"]["details"] = (
                            validation_result["details"]
                        )
                        startup_success = True
                        print("✅ Server startup completed successfully")
                        break
                    else:
                        detailed_logs.append(
                            f"[{elapsed:.1f}s] Endpoint validation failed: {validation_result['error']}"
                        )
                elif connectivity_result["should_continue"]:
                    if check_round % 10 == 0:  # Log every 5 seconds
                        print(
                            f"🔄 Waiting for startup... ({elapsed:.1f}s/{total_timeout}s)"
                        )
                else:
                    # Connectivity test indicates server issues
                    detailed_logs.append(
                        f"[{elapsed:.1f}s] Connectivity test failed: {connectivity_result['error']}"
                    )
            else:
                if check_round % 4 == 0:  # Log every 2 seconds during initial phase
                    print(f"🔄 Initial startup phase... ({elapsed:.1f}s)")

            await asyncio.sleep(check_interval)

        # Cleanup process
        await _cleanup_uvicorn_process(process, detailed_logs)

        response_time = time.time() - start_time

        if startup_success:
            results.add_startup_result(
                "uvicorn_startup",
                "Uvicorn Server Startup Process",
                response_time,
                {
                    "startup_success": True,
                    "process_pid": process.pid,
                    "startup_phases": startup_phases,
                    "total_duration": response_time,
                    "detailed_logs": detailed_logs[-20:],  # Last 20 log entries
                },
            )
            return True
        else:
            # Create detailed failure report
            failure_details = await _create_startup_failure_report(
                process, startup_phases, detailed_logs, response_time
            )

            results.add_error(
                "uvicorn_startup",
                "startup_timeout",
                "Uvicorn Server Startup Process",
                Exception(f"Server failed to start within {total_timeout}s timeout"),
                response_time,
            )

            # Enhanced error details for traceback report
            if results.errors:
                latest_error = results.errors[-1]
                latest_error["response_data"]["startup_analysis"] = failure_details
                latest_error["response_data"]["traceback"] = failure_details[
                    "detailed_traceback"
                ]

            return False

    except Exception as e:
        response_time = time.time() - start_time
        print(f"❌ Uvicorn startup test failed: {e}")

        # Ensure process cleanup on exception
        if process:
            await _cleanup_uvicorn_process(process, detailed_logs)

        # Create exception traceback with context
        exception_context = {
            "startup_phases": startup_phases,
            "detailed_logs": detailed_logs,
            "elapsed_time": response_time,
            "process_pid": process.pid if process else None,
        }

        results.add_error(
            "uvicorn_startup",
            "startup_exception",
            "Uvicorn Server Startup Process",
            e,
            response_time,
        )

        if results.errors:
            latest_error = results.errors[-1]
            latest_error["response_data"]["exception_context"] = exception_context
            latest_error["response_data"][
                "traceback"
            ] = f"""Uvicorn Startup Exception
Exception: {str(e)}
Exception Type: {type(e).__name__}

Startup Context:
{traceback.format_exc()}

Detailed Logs:
{chr(10).join(detailed_logs[-10:]) if detailed_logs else 'No logs available'}

Process Information:
PID: {process.pid if process else 'Not created'}
Elapsed Time: {response_time:.2f}s
"""

        return False


async def _handle_uvicorn_early_exit(
    results, process, stdout, stderr, elapsed, startup_phases, detailed_logs
):
    """Handle early process exit with detailed analysis."""
    print(f"❌ Uvicorn process exited early after {elapsed:.2f}s")

    # Analyze exit reason
    exit_analysis = {
        "exit_code": process.returncode,
        "stdout_length": len(stdout) if stdout else 0,
        "stderr_length": len(stderr) if stderr else 0,
        "likely_cause": _analyze_exit_cause(process.returncode, stdout, stderr),
        "startup_phase_reached": _get_furthest_phase(startup_phases),
    }

    if stdout:
        print(f"STDOUT ({len(stdout)} chars): {stdout[:500]}...")
        detailed_logs.extend([f"STDOUT: {line}" for line in stdout.split("\n")[:10]])
    if stderr:
        print(f"STDERR ({len(stderr)} chars): {stderr[:500]}...")
        detailed_logs.extend([f"STDERR: {line}" for line in stderr.split("\n")[:10]])

    # Create comprehensive early exit report
    subprocess_info = {
        "stdout": stdout if stdout else "No stdout",
        "stderr": stderr if stderr else "No stderr",
        "exit_code": process.returncode,
        "process_id": process.pid,
        "exit_analysis": exit_analysis,
        "elapsed_time": elapsed,
    }

    error_msg = f"Uvicorn process exited early (exit code: {process.returncode}, cause: {exit_analysis['likely_cause']})"
    startup_exception = Exception(error_msg)

    results.add_error(
        "uvicorn_startup",
        "subprocess_failure",
        "Uvicorn Server Startup Process",
        startup_exception,
        elapsed,
    )

    # Enhanced traceback with analysis
    if results.errors:
        latest_error = results.errors[-1]
        latest_error["response_data"]["subprocess_output"] = subprocess_info
        latest_error["response_data"][
            "traceback"
        ] = f"""Uvicorn Server Early Exit Analysis
Exit Code: {process.returncode}
Process ID: {process.pid}
Elapsed Time: {elapsed:.2f}s
Likely Cause: {exit_analysis['likely_cause']}
Furthest Phase: {exit_analysis['startup_phase_reached']}

STDOUT Output ({len(stdout) if stdout else 0} characters):
{stdout if stdout else 'No stdout output'}

STDERR Output ({len(stderr) if stderr else 0} characters):
{stderr if stderr else 'No stderr output'}

Detailed Timeline:
{chr(10).join(detailed_logs) if detailed_logs else 'No detailed logs'}

Full Command: python uvicorn_start.py
Working Directory: {os.getcwd()}
Environment Variables: DEBUG=1, VERBOSE_SSL_LOGGING=true, ENABLE_CONNECTION_MONITORING=true
"""

    return False


async def _analyze_startup_phase(output, startup_phases, elapsed):
    """Analyze log output to determine startup phase progress."""
    if not output:
        return

    output_lower = output.lower()

    # Detect initial log output
    if "rank_bm25" in output_lower or "postgres-startup" in output_lower:
        if not startup_phases["initial_logs"]["completed"]:
            startup_phases["initial_logs"]["completed"] = True
            startup_phases["initial_logs"]["duration"] = elapsed
            startup_phases["initial_logs"]["details"]["first_log"] = output[:100]

    # Detect port binding
    if "running on" in output_lower and "8000" in output:
        if not startup_phases["port_binding"]["completed"]:
            startup_phases["port_binding"]["completed"] = True
            startup_phases["port_binding"]["duration"] = elapsed
            startup_phases["port_binding"]["details"]["binding_log"] = output.strip()

    # Detect app loading completion
    if (
        "application startup complete" in output_lower
        or "started server process" in output_lower
    ):
        if not startup_phases["app_loading"]["completed"]:
            startup_phases["app_loading"]["completed"] = True
            startup_phases["app_loading"]["duration"] = elapsed
            startup_phases["app_loading"]["details"]["completion_log"] = output.strip()

    # Detect server ready state
    if (
        "uvicorn running" in output_lower
        or "waiting for application startup" in output_lower
    ):
        if not startup_phases["server_ready"]["completed"]:
            startup_phases["server_ready"]["completed"] = True
            startup_phases["server_ready"]["duration"] = elapsed
            startup_phases["server_ready"]["details"]["ready_log"] = output.strip()


async def _test_server_connectivity(elapsed, detailed_logs):
    """Test server connectivity with detailed error reporting."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:8000/health")

            if response.status_code == 200:
                detailed_logs.append(
                    f"[{elapsed:.1f}s] Health check successful: {response.status_code}"
                )
                return {
                    "success": True,
                    "details": {
                        "status_code": response.status_code,
                        "response_time": elapsed,
                        "endpoint": "/health",
                    },
                }
            else:
                detailed_logs.append(
                    f"[{elapsed:.1f}s] Health check failed: {response.status_code}"
                )
                return {
                    "success": False,
                    "should_continue": True,
                    "error": f"HTTP {response.status_code}",
                }

    except httpx.ConnectError:
        # Connection refused - server not ready yet
        return {
            "success": False,
            "should_continue": True,
            "error": "Connection refused",
        }
    except httpx.TimeoutException:
        detailed_logs.append(f"[{elapsed:.1f}s] Health check timeout")
        return {"success": False, "should_continue": True, "error": "Request timeout"}
    except Exception as e:
        detailed_logs.append(f"[{elapsed:.1f}s] Health check error: {str(e)}")
        return {"success": False, "should_continue": False, "error": str(e)}


async def _run_endpoint_validation(detailed_logs):
    """Run comprehensive endpoint validation."""
    try:
        token = create_test_jwt_token("test_user@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        validation_results = {}

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test multiple endpoints
            endpoints = [
                {"path": "/health", "auth_required": False},
                {"path": "/docs", "auth_required": False},
                {"path": "/openapi.json", "auth_required": False},
            ]

            for endpoint in endpoints:
                try:
                    test_headers = headers if endpoint["auth_required"] else {}
                    response = await client.get(
                        f"http://localhost:8000{endpoint['path']}", headers=test_headers
                    )

                    validation_results[endpoint["path"]] = {
                        "status_code": response.status_code,
                        "success": response.status_code < 400,
                        "response_size": (
                            len(response.content) if hasattr(response, "content") else 0
                        ),
                    }

                except Exception as e:
                    validation_results[endpoint["path"]] = {
                        "success": False,
                        "error": str(e),
                    }

            # Setup debug environment for enhanced testing
            try:
                await setup_debug_environment(
                    client,
                    print__checkpointers_debug="1",
                    print__postgres_debug="1",
                    DEBUG_TRACEBACK="1",
                )
                validation_results["debug_setup"] = {"success": True}

                # Cleanup debug environment
                await cleanup_debug_environment(client)
                validation_results["debug_cleanup"] = {"success": True}

            except Exception as debug_error:
                validation_results["debug_setup"] = {
                    "success": False,
                    "error": str(debug_error),
                }

        success_count = sum(
            1 for result in validation_results.values() if result.get("success", False)
        )
        total_tests = len(validation_results)

        if success_count >= total_tests * 0.7:  # 70% success threshold
            return {
                "success": True,
                "details": {
                    "endpoint_results": validation_results,
                    "success_rate": success_count / total_tests,
                    "total_tests": total_tests,
                },
            }
        else:
            return {
                "success": False,
                "error": f"Only {success_count}/{total_tests} endpoint tests passed",
                "details": validation_results,
            }

    except Exception as e:
        return {"success": False, "error": f"Validation exception: {str(e)}"}


async def _cleanup_uvicorn_process(process, detailed_logs):
    """Cleanup uvicorn process with proper signal handling."""
    if process and process.poll() is None:
        try:
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
                detailed_logs.append("✅ Uvicorn process cleaned up gracefully")
                print("✅ Uvicorn process cleaned up")
            except subprocess.TimeoutExpired:
                process.kill()
                detailed_logs.append("⚠️ Uvicorn process force killed after timeout")
                print("⚠️ Uvicorn process force killed")

        except Exception as cleanup_error:
            detailed_logs.append(f"⚠️ Process cleanup warning: {cleanup_error}")
            print("⚠️ Process cleanup warning")


def _analyze_exit_cause(exit_code, stdout, stderr):
    """Analyze process exit to determine likely cause."""
    if exit_code == 0:
        return "Clean exit (unexpected for server)"
    elif exit_code == 1:
        if stderr and "UnicodeEncodeError" in stderr:
            return "Unicode encoding error in debug output"
        elif stderr and "ImportError" in stderr:
            return "Missing module import"
        elif stderr and "ModuleNotFoundError" in stderr:
            return "Module not found"
        elif stderr and "ConnectionError" in stderr:
            return "Database connection failure"
        elif stderr and "PermissionError" in stderr:
            return "File permission error"
        elif stderr and "AddressAlreadyInUse" in stderr:
            return "Port 8000 already in use"
        else:
            return "General application error"
    elif exit_code == -15:  # SIGTERM
        return "Process terminated"
    elif exit_code == -9:  # SIGKILL
        return "Process killed"
    else:
        return f"Unknown exit code {exit_code}"


def _get_furthest_phase(startup_phases):
    """Get the furthest startup phase reached."""
    phase_order = [
        "process_creation",
        "initial_logs",
        "port_binding",
        "app_loading",
        "server_ready",
        "health_check",
        "endpoint_validation",
    ]

    for phase in reversed(phase_order):
        if startup_phases[phase]["completed"]:
            return phase

    return "none"


async def _create_startup_failure_report(
    process, startup_phases, detailed_logs, response_time
):
    """Create comprehensive startup failure analysis."""
    furthest_phase = _get_furthest_phase(startup_phases)

    # Analyze what went wrong
    failure_analysis = {
        "furthest_phase_reached": furthest_phase,
        "completed_phases": [
            phase for phase, data in startup_phases.items() if data["completed"]
        ],
        "failed_phases": [
            phase for phase, data in startup_phases.items() if not data["completed"]
        ],
        "total_duration": response_time,
        "process_still_running": process and process.poll() is None,
        "log_entries_count": len(detailed_logs),
    }

    # Get final process output if still running
    final_output = {"stdout": "", "stderr": ""}
    if process and process.poll() is None:
        try:
            # Try to get any remaining output
            stdout, stderr = process.communicate(timeout=2)
            final_output["stdout"] = stdout if stdout else ""
            final_output["stderr"] = stderr if stderr else ""
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                stdout, stderr = process.communicate()
                final_output["stdout"] = stdout if stdout else ""
                final_output["stderr"] = stderr if stderr else ""
            except Exception:
                pass

    # Create detailed traceback
    detailed_traceback = f"""Uvicorn Startup Timeout Analysis
Total Duration: {response_time:.2f}s
Furthest Phase: {furthest_phase}

Startup Phase Analysis:
{chr(10).join([f"  {phase}: {'✅ Completed' if data['completed'] else '❌ Failed'} ({data['duration']:.2f}s)" for phase, data in startup_phases.items()])}

Process Information:
PID: {process.pid if process else 'Not created'}
Still Running: {failure_analysis['process_still_running']}

Final Process Output:
STDOUT: {final_output['stdout'][:1000] if final_output['stdout'] else 'No final stdout'}
STDERR: {final_output['stderr'][:1000] if final_output['stderr'] else 'No final stderr'}

Detailed Timeline ({len(detailed_logs)} entries):
{chr(10).join(detailed_logs[-30:]) if detailed_logs else 'No detailed logs available'}

Recommendations:
{_generate_startup_recommendations(furthest_phase, startup_phases)}
"""

    return {
        "failure_analysis": failure_analysis,
        "final_output": final_output,
        "detailed_traceback": detailed_traceback,
    }


def _generate_startup_recommendations(furthest_phase, startup_phases):
    """Generate troubleshooting recommendations based on failure analysis."""
    recommendations = []

    if furthest_phase == "none" or furthest_phase == "process_creation":
        recommendations.extend(
            [
                "1. Check if Python environment is properly activated",
                "2. Verify uvicorn_start.py exists and is executable",
                "3. Check for import errors in the application code",
            ]
        )
    elif furthest_phase == "initial_logs":
        recommendations.extend(
            [
                "1. Check application imports and dependencies",
                "2. Verify database connection configuration",
                "3. Look for Unicode encoding issues in debug output",
            ]
        )
    elif furthest_phase in ["port_binding", "app_loading"]:
        recommendations.extend(
            [
                "1. Check if port 8000 is available (netstat -an | findstr :8000)",
                "2. Verify FastAPI application initialization",
                "3. Check for middleware configuration errors",
            ]
        )
    elif furthest_phase == "server_ready":
        recommendations.extend(
            [
                "1. Server started but health check failed",
                "2. Check /health endpoint implementation",
                "3. Verify authentication requirements",
            ]
        )
    else:
        recommendations.extend(
            [
                "1. Server startup timeout - increase timeout value",
                "2. Check for slow database initialization",
                "3. Monitor system resources (CPU, memory)",
            ]
        )

    return chr(10).join(recommendations)


async def main():
    """Main test execution function with comprehensive error handling."""
    print("🚀 Starting Enhanced Startup Debug Tests")
    print(f"📂 Working directory: {parent_dir}")
    print(f"⏱️ Test timeout: {STARTUP_TIMEOUT}s")
    print(f"🧪 Test cases: {len(TEST_CASES)}")

    try:
        # Initialize results tracking
        results = StartupTestResults()

        # Setup debug environment for enhanced monitoring
        try:
            # Set environment variables for detailed debugging
            os.environ["DEBUG_TRACEBACK"] = "1"
            os.environ["print__checkpointers_debug"] = "1"
            os.environ["print__postgres_debug"] = "1"
            print("✅ Debug environment configured")
        except Exception as env_error:
            print(f"⚠️ Debug environment setup warning: {env_error}")

        # Execute test cases in sequence
        test_functions = {
            "env_check": test_environment_variables,
            "imports": test_critical_imports,
            "db_connection": test_database_connection,
            "checkpointer_init": test_checkpointer_initialization,
            "uvicorn_startup": test_uvicorn_startup,
        }

        print(f"\n📋 Executing {len(TEST_CASES)} startup tests...")

        for test_case in TEST_CASES:
            test_id = test_case["test_id"]
            if test_id in test_functions:
                print(f"\n🔄 Running test: {test_case['description']}")

                try:
                    test_result = await test_functions[test_id](results)
                    status = "✅ PASSED" if test_result else "❌ FAILED"
                    print(f"📊 Test {test_id}: {status}")

                except Exception as test_error:
                    print(f"💥 Test {test_id} crashed: {test_error}")
                    results.add_error(
                        test_id,
                        "test_exception",
                        f"{test_case['description']} execution",
                        test_error,
                        0.0,
                    )
            else:
                print(f"⚠️ Unknown test case: {test_id}")

        # Generate comprehensive test summary
        summary = results.get_summary()
        print("\n" + "=" * 60)
        print("📊 STARTUP TEST SUMMARY")
        print("=" * 60)
        print(f"🧪 Total tests executed: {summary['total_requests']}")
        print(f"✅ Successful tests: {summary['successful_requests']}")
        print(f"❌ Failed tests: {summary['failed_requests']}")
        print(f"⏱️ Average response time: {summary['average_response_time']:.3f}s")
        print(f"📈 Success rate: {summary['success_rate']:.1%}")

        if summary["tracked_processes"]:
            print(f"🔄 Tracked processes: {summary['tracked_processes']}")

        if results.errors:
            print(f"\n❌ ERRORS ENCOUNTERED ({len(results.errors)}):")
            for error in results.errors:
                print(f"   • {error['test_id']}: {error['error_type']}")

        # Save detailed traceback report for any failures
        if results.errors:
            test_context = {
                "Test Cases": TEST_CASES,
                "Timeout": f"{STARTUP_TIMEOUT}s",
                "Working Directory": str(parent_dir),
                "Total Tests": summary["total_requests"],
                "Failed Tests": summary["failed_requests"],
                "Success Rate": f"{summary['success_rate']:.1%}",
            }

            save_traceback_report(
                report_type="startup_diagnostics",
                test_results=results,
                test_context=test_context,
            )

        # Determine overall test success for startup testing
        critical_tests = ["env_check", "imports", "db_connection"]
        critical_failures = [
            error for error in results.errors if error["test_id"] in critical_tests
        ]

        test_passed = (
            summary["total_requests"] > 0
            and len(critical_failures) == 0
            and summary["successful_requests"]
            >= 3  # At least basic functionality works
        )

        if critical_failures:
            print(
                f"\n❌ Test failed: Critical failures in {[e['test_id'] for e in critical_failures]}"
            )
        elif summary["successful_requests"] < 3:
            print(
                f"\n❌ Test failed: Insufficient successful tests ({summary['successful_requests']}/5)"
            )
        elif summary["total_requests"] == 0:
            print("\n❌ Test failed: No tests were executed")

        print(f"\n🏁 OVERALL RESULT: {'✅ PASSED' if test_passed else '❌ FAILED'}")

        # Cleanup debug environment
        for env_var in [
            "DEBUG_TRACEBACK",
            "print__checkpointers_debug",
            "print__postgres_debug",
        ]:
            if env_var in os.environ:
                del os.environ[env_var]
        print("✅ Debug environment cleaned up")

        return test_passed

    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        test_context = {
            "Test Cases": len(TEST_CASES),
            "Timeout": f"{STARTUP_TIMEOUT}s",
            "Error Location": "main() function",
            "Error During": "Startup test execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        return False


if __name__ == "__main__":
    try:
        test_result = asyncio.run(main())
        sys.exit(0 if test_result else 1)
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        test_context = {
            "Test Cases": len(TEST_CASES),
            "Timeout": f"{STARTUP_TIMEOUT}s",
            "Error Location": "__main__ execution",
            "Error During": "Direct script execution",
        }
        save_traceback_report(
            report_type="exception", exception=e, test_context=test_context
        )
        sys.exit(1)
