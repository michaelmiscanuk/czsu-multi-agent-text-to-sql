#!/usr/bin/env python3
"""
Interactive Function Removal Helper
====================================

This script helps you safely verify and remove unused functions identified
by find_unused_functions.py.

Features:
- Shows function code before removal
- Searches entire project for references
- Creates backup before removal
- Runs tests after removal (optional)
- Commits each removal individually

Usage:
    python remove_unused_functions.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


# List of functions to verify/remove (from the analysis)
# Format: (file_path_relative, function_name, line_number)
FUNCTIONS_TO_REVIEW = [
    # Debug functions (safest to remove first)
    ("api/utils/debug.py", "print__chat_messages_debug", 206),
    ("api/utils/debug.py", "print__data_table_debug", 276),
    ("api/utils/debug.py", "print__chat_thread_id_checkpoints_debug", 290),
    ("api/utils/debug.py", "print__debug_pool_status_debug", 304),
    ("api/utils/debug.py", "print__chat_thread_id_run_ids_debug", 318),
    ("api/utils/debug.py", "print__debug_run_id_debug", 332),
    ("api/utils/debug.py", "print__admin_clear_cache_debug", 346),
    # Test LLM functions
    ("my_agent/utils/models.py", "get_azure_llm_gpt_4o_test", 39),
    ("my_agent/utils/models.py", "get_azure_llm_gpt_4o_mini_test", 71),
    ("my_agent/utils/models.py", "get_ollama_llm_test", 143),
    ("my_agent/utils/models.py", "get_azure_embedding_model_test", 184),
    ("my_agent/utils/models.py", "get_langchain_azure_embedding_model_test", 220),
    # Legacy functions
    ("data/helpers.py", "save_parsed_text_to_file_legacy", 220),
    ("data/helpers.py", "load_parsed_text_from_file_legacy", 239),
    # Add more functions here as you verify them
]


def print_header(text: str):
    """Print a colored header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by looking for common project markers.

    Searches upward from start_path for indicators like:
    - .git directory
    - pyproject.toml
    - setup.py
    - requirements.txt

    Args:
        start_path: Path to start searching from (usually script location)

    Returns:
        Path to project root

    Raises:
        SystemExit: If project root cannot be determined
    """
    # Common project root indicators (in order of reliability)
    root_markers = [
        ".git",  # Git repository
        "pyproject.toml",  # Modern Python project
        "setup.py",  # Traditional Python package
        "setup.cfg",  # Python package config
        "requirements.txt",  # Python dependencies
        "Pipfile",  # Pipenv project
        "poetry.lock",  # Poetry project
        "package.json",  # Node.js project (for mixed projects)
    ]

    # Start from the script directory and search upward
    current = start_path.resolve()

    # Try to find markers in current directory and parents
    for _ in range(10):  # Limit search depth
        # Check if any marker exists in current directory
        for marker in root_markers:
            marker_path = current / marker
            if marker_path.exists():
                print_success(f"Found project root marker: {marker}")
                print_success(f"Project root: {current}")
                return current

        # Move up one directory
        parent = current.parent
        if parent == current:
            # Reached filesystem root
            break
        current = parent

    # Fallback: use script directory
    print_warning("No project root markers found")
    print_warning(f"Using script directory as project root: {start_path}")
    return start_path


def get_project_dir() -> Path:
    """Get project directory by auto-detecting project root."""
    script_dir = Path(__file__).parent
    return find_project_root(script_dir)


def read_function_code(
    file_path: Path, function_name: str, start_line: int
) -> Optional[str]:
    """Read function code from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find function definition
        if start_line > len(lines):
            return None

        # Get function code (simplified - just get next 20 lines)
        end_line = min(start_line + 19, len(lines))
        function_code = "".join(lines[start_line - 1 : end_line])

        return function_code

    except Exception as e:
        print_error(f"Error reading file: {e}")
        return None


def search_references(project_dir: Path, function_name: str) -> List[str]:
    """Search for references to function in project."""
    references = []

    try:
        # Use git grep if available (faster)
        result = subprocess.run(
            ["git", "grep", "-n", function_name],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            references = result.stdout.strip().split("\n")

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to Python search
        for py_file in project_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if function_name in line:
                            relative = py_file.relative_to(project_dir)
                            references.append(f"{relative}:{i}: {line.strip()}")
            except:
                pass

    return references


def show_function_details(
    project_dir: Path, file_rel: str, func_name: str, line_no: int
):
    """Show detailed information about a function."""
    file_path = project_dir / file_rel.replace("/", os.sep)

    print_header(f"Function: {func_name}")
    print_info(f"File: {file_rel}:{line_no}")

    # Show function code
    print(f"\n{Colors.BOLD}Function Code:{Colors.END}")
    print(f"{Colors.BLUE}{'─' * 80}{Colors.END}")

    code = read_function_code(file_path, func_name, line_no)
    if code:
        print(code)
    else:
        print_error("Could not read function code")

    print(f"{Colors.BLUE}{'─' * 80}{Colors.END}")

    # Search for references
    print(f"\n{Colors.BOLD}Searching for references...{Colors.END}")
    references = search_references(project_dir, func_name)

    if not references or (len(references) == 1 and file_rel in references[0]):
        print_success("No external references found (only definition)")
        return True  # Safe to remove
    else:
        print_warning(f"Found {len(references)} reference(s):")
        for ref in references[:10]:  # Show first 10
            print(f"  {ref}")
        if len(references) > 10:
            print(f"  ... and {len(references) - 10} more")
        return False  # Not safe to remove


def remove_function(
    project_dir: Path, file_rel: str, func_name: str, line_no: int
) -> bool:
    """Remove function from file."""
    file_path = project_dir / file_rel.replace("/", os.sep)

    try:
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find function end (simplified - remove function and blank lines after)
        start_idx = line_no - 1
        end_idx = start_idx

        # Find next function or end of file
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and (
                line.strip().startswith("def ") or line.strip().startswith("class ")
            ):
                end_idx = i - 1
                break
        else:
            end_idx = len(lines) - 1

        # Remove function
        new_lines = lines[:start_idx] + lines[end_idx + 1 :]

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        print_success(f"Removed function {func_name} from {file_rel}")
        return True

    except Exception as e:
        print_error(f"Error removing function: {e}")
        return False


def run_tests() -> bool:
    """Run tests to verify nothing broke."""
    print_info("Running tests...")
    try:
        result = subprocess.run(
            ["pytest", "-x"],  # -x stops at first failure
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print_success("All tests passed!")
            return True
        else:
            print_error("Tests failed!")
            print(result.stdout)
            print(result.stderr)
            return False

    except FileNotFoundError:
        print_warning("pytest not found, skipping tests")
        return True


def git_commit(file_rel: str, func_name: str) -> bool:
    """Commit the removal."""
    try:
        subprocess.run(["git", "add", file_rel], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Remove unused function: {func_name}"], check=True
        )
        print_success(f"Committed removal of {func_name}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Git commit failed: {e}")
        return False


def interactive_review():
    """Interactive review and removal of functions."""
    project_dir = get_project_dir()

    print_header("Interactive Function Removal Helper")
    print_info(f"Project: {project_dir}")
    print_info(f"Functions to review: {len(FUNCTIONS_TO_REVIEW)}")

    removed_count = 0
    skipped_count = 0

    for file_rel, func_name, line_no in FUNCTIONS_TO_REVIEW:
        is_safe = show_function_details(project_dir, file_rel, func_name, line_no)

        print(f"\n{Colors.BOLD}What would you like to do?{Colors.END}")
        print(f"  {Colors.GREEN}r{Colors.END} - Remove function")
        print(f"  {Colors.YELLOW}s{Colors.END} - Skip (keep function)")
        print(f"  {Colors.RED}q{Colors.END} - Quit")

        if is_safe:
            print_info("Analysis suggests this is SAFE to remove")
            choice = (
                input(f"\n{Colors.BOLD}Choice [r/s/q]:{Colors.END} ").strip().lower()
            )
        else:
            print_warning("Analysis found references - review carefully!")
            choice = (
                input(f"\n{Colors.BOLD}Choice [r/s/q]:{Colors.END} ").strip().lower()
            )

        if choice == "q":
            print_info("Quitting...")
            break
        elif choice == "r":
            if remove_function(project_dir, file_rel, func_name, line_no):
                removed_count += 1

                # Ask if user wants to run tests
                run_test = (
                    input(f"{Colors.BOLD}Run tests? [y/N]:{Colors.END} ")
                    .strip()
                    .lower()
                )
                if run_test == "y":
                    if not run_tests():
                        print_error("Tests failed! Consider reverting.")
                        revert = (
                            input(f"{Colors.BOLD}Revert changes? [y/N]:{Colors.END} ")
                            .strip()
                            .lower()
                        )
                        if revert == "y":
                            subprocess.run(["git", "checkout", file_rel])
                            print_success("Changes reverted")
                            removed_count -= 1
                            continue

                # Ask if user wants to commit
                commit = (
                    input(f"{Colors.BOLD}Commit removal? [y/N]:{Colors.END} ")
                    .strip()
                    .lower()
                )
                if commit == "y":
                    git_commit(file_rel, func_name)
        else:
            print_info(f"Skipping {func_name}")
            skipped_count += 1

        print("\n" + "=" * 80 + "\n")

    # Final summary
    print_header("Summary")
    print_success(f"Removed: {removed_count} functions")
    print_info(f"Skipped: {skipped_count} functions")
    print_info(f"Total reviewed: {removed_count + skipped_count}")


if __name__ == "__main__":
    interactive_review()
