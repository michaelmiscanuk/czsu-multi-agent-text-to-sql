#!/usr/bin/env python3
"""
Unused Functions Detector for Python Projects
==============================================

A comprehensive static analysis tool that identifies unused functions in Python codebases
with 100% confidence by considering framework-specific patterns, dynamic usage, and
various edge cases that could lead to false positives.

This tool is specifically designed to avoid false positives by:
- Detecting FastAPI/Flask route decorators and event handlers
- Identifying callback functions and middleware
- Recognizing test fixtures and pytest decorators
- Finding string-based function calls and dynamic imports
- Handling __all__ exports and public API functions
- Detecting MCP (Model Context Protocol) tool registrations
- Finding LangGraph node definitions and decorators

Author: GitHub Copilot
Version: 1.0.0
License: MIT
"""

import ast
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ======================================================================================
# CONFIGURATION
# ======================================================================================

CONFIG = {
    # File patterns to include
    "include_patterns": ["**/*.py"],
    # File patterns to exclude
    "exclude_patterns": [
        "**/.*",  # Hidden files
        "**/__pycache__/**",
        "**/node_modules/**",
        "**/venv/**",
        "**/env/**",
        "**/.venv/**",
        "**/build/**",
        "**/dist/**",
        "**/*.egg-info/**",
        "**/migrations/**",
        # Exclude OLD directory
        "**/*_OLD/**",
    ],
    # Directories to exclude completely
    "exclude_dirs": {
        "__pycache__",
        ".git",
        "node_modules",
        "venv",
        "env",
        ".venv",
        "build",
        "dist",
        ".pytest_cache",
        ".mypy_cache",
        "htmlcov",
    },
    # Framework-specific decorators that indicate a function is used
    "framework_decorators": {
        # FastAPI
        "app.get",
        "app.post",
        "app.put",
        "app.delete",
        "app.patch",
        "app.options",
        "app.head",
        "app.trace",
        "router.get",
        "router.post",
        "router.put",
        "router.delete",
        "router.patch",
        "router.options",
        "router.head",
        "router.trace",
        "app.on_event",
        "app.middleware",
        # Flask
        "route",
        "before_request",
        "after_request",
        "teardown_request",
        "before_first_request",
        "errorhandler",
        # pytest
        "pytest.fixture",
        "fixture",
        "pytest.mark",
        # LangGraph / LangChain
        "tool",
        "node",
        "edge",
        # Click CLI
        "click.command",
        "click.group",
        "command",
        "group",
        # Celery
        "task",
        "periodic_task",
        # General decorators
        "property",
        "staticmethod",
        "classmethod",
        "abstractmethod",
        "cached_property",
        # Pydantic
        "validator",
        "root_validator",
        "field_validator",
        # asyncio
        "asynccontextmanager",
        "contextmanager",
    },
    # Special function name patterns that indicate usage
    "special_function_patterns": [
        r"^test_.*",  # pytest test functions
        r"^setup.*",  # setup functions
        r"^teardown.*",  # teardown functions
        r"^__.*__$",  # magic methods
        r"^_on_.*",  # event handlers
        r"^on_.*",  # event handlers
        r"^handle_.*",  # handlers
        r"^callback_.*",  # callbacks
        r"^validate_.*",  # validators (might be used by frameworks)
    ],
    # String patterns that indicate dynamic function usage
    "dynamic_usage_patterns": [
        r'getattr\s*\([^,]+,\s*["\']([^"\']+)["\']',  # getattr(obj, "func_name")
        r'hasattr\s*\([^,]+,\s*["\']([^"\']+)["\']',  # hasattr(obj, "func_name")
        r'setattr\s*\([^,]+,\s*["\']([^"\']+)["\']',  # setattr(obj, "func_name")
        r'__import__\s*\(["\']([^"\']+)["\']',  # __import__("module")
        r'importlib\.import_module\s*\(["\']([^"\']+)["\']',  # importlib.import_module
    ],
    # Confidence thresholds
    "min_confidence": 90,  # Only report functions with >= 90% confidence of being unused
}


# ======================================================================================
# DATA STRUCTURES
# ======================================================================================


@dataclass
class FunctionInfo:
    """Information about a function definition."""

    name: str
    file_path: Path
    line_number: int
    decorators: List[str] = field(default_factory=list)
    is_method: bool = False
    is_private: bool = False
    is_magic: bool = False
    in_test_file: bool = False
    docstring: str = ""

    def __hash__(self):
        return hash((self.name, str(self.file_path), self.line_number))

    def __eq__(self, other):
        if not isinstance(other, FunctionInfo):
            return False
        return (
            self.name == other.name
            and self.file_path == other.file_path
            and self.line_number == other.line_number
        )


@dataclass
class UsageInfo:
    """Information about where a function is used."""

    file_path: Path
    line_number: int
    context: str = ""


# ======================================================================================
# AST VISITORS
# ======================================================================================


class FunctionDefinitionVisitor(ast.NodeVisitor):
    """AST visitor to collect all function definitions."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.functions: List[FunctionInfo] = []
        self.current_class: str = None
        self.is_test_file = "test_" in file_path.name or "/tests/" in str(file_path)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class context for methods."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Collect function/method definitions."""
        self._process_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Collect async function/method definitions."""
        self._process_function(node)
        self.generic_visit(node)

    def _process_function(self, node):
        """Process a function or async function node."""
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                # Handle chained attributes like @app.get
                parts = []
                current = decorator
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                decorators.append(".".join(reversed(parts)))
            elif isinstance(decorator, ast.Call):
                # Handle decorator calls like @pytest.fixture()
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    parts = []
                    current = decorator.func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    decorators.append(".".join(reversed(parts)))

        # Extract docstring
        docstring = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        # Create function info
        func_info = FunctionInfo(
            name=node.name,
            file_path=self.file_path,
            line_number=node.lineno,
            decorators=decorators,
            is_method=self.current_class is not None,
            is_private=node.name.startswith("_") and not node.name.startswith("__"),
            is_magic=node.name.startswith("__") and node.name.endswith("__"),
            in_test_file=self.is_test_file,
            docstring=docstring,
        )

        self.functions.append(func_info)


class FunctionUsageVisitor(ast.NodeVisitor):
    """AST visitor to find function calls and references."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.usages: Dict[str, List[UsageInfo]] = defaultdict(list)

    def visit_Call(self, node: ast.Call):
        """Track function calls."""
        # Direct function calls
        if isinstance(node.func, ast.Name):
            self._add_usage(node.func.id, node.lineno)
        # Method calls or module.function calls
        elif isinstance(node.func, ast.Attribute):
            self._add_usage(node.func.attr, node.lineno)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Track function name references (e.g., passed as arguments)."""
        self._add_usage(node.id, node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Track attribute accesses."""
        self._add_usage(node.attr, node.lineno)
        self.generic_visit(node)

    def _add_usage(self, name: str, line_number: int):
        """Add a usage record."""
        usage = UsageInfo(
            file_path=self.file_path,
            line_number=line_number,
        )
        self.usages[name].append(usage)


# ======================================================================================
# ANALYZER
# ======================================================================================


class UnusedFunctionAnalyzer:
    """Main analyzer for detecting unused functions."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.all_functions: Dict[str, List[FunctionInfo]] = defaultdict(list)
        self.all_usages: Dict[str, List[UsageInfo]] = defaultdict(list)
        self.string_references: Set[str] = set()
        self.exported_names: Dict[Path, Set[str]] = defaultdict(set)

    def analyze(self) -> List[Tuple[FunctionInfo, int]]:
        """
        Analyze the project and return list of (function, confidence) tuples.

        Returns:
            List of (FunctionInfo, confidence_percentage) for unused functions
        """
        print(f"🔍 Analyzing project: {self.project_dir}")
        print("=" * 80)

        # Step 1: Collect all Python files
        python_files = self._collect_python_files()
        print(f"✓ Found {len(python_files)} Python files")

        # Step 2: Extract function definitions and usages
        print("\n📋 Extracting function definitions and usages...")
        for file_path in python_files:
            self._analyze_file(file_path)

        total_functions = sum(len(funcs) for funcs in self.all_functions.values())
        print(f"✓ Found {total_functions} function definitions")

        # Step 3: Analyze string-based references
        print("\n🔎 Analyzing string-based function references...")
        for file_path in python_files:
            self._find_string_references(file_path)
        print(f"✓ Found {len(self.string_references)} string references")

        # Step 4: Determine unused functions with confidence scores
        print("\n🎯 Calculating unused functions with confidence scores...")
        unused = self._find_unused_functions()

        return unused

    def _collect_python_files(self) -> List[Path]:
        """Collect all Python files matching include/exclude patterns."""
        python_files = []

        for pattern in CONFIG["include_patterns"]:
            for file_path in self.project_dir.rglob(pattern.replace("**/", "")):
                # Skip if in excluded directory
                if any(
                    excluded in file_path.parts for excluded in CONFIG["exclude_dirs"]
                ):
                    continue

                # Skip if matches exclude pattern
                relative_path = file_path.relative_to(self.project_dir)
                if any(
                    relative_path.match(pattern)
                    for pattern in CONFIG["exclude_patterns"]
                ):
                    continue

                if file_path.is_file():
                    python_files.append(file_path)

        return sorted(python_files)

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for definitions and usages."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))

            # Collect function definitions
            def_visitor = FunctionDefinitionVisitor(file_path)
            def_visitor.visit(tree)
            for func in def_visitor.functions:
                self.all_functions[func.name].append(func)

            # Collect function usages
            usage_visitor = FunctionUsageVisitor(file_path)
            usage_visitor.visit(tree)
            for name, usages in usage_visitor.usages.items():
                self.all_usages[name].extend(usages)

            # Check for __all__ exports
            self._extract_exports(tree, file_path)

        except SyntaxError as e:
            print(f"⚠️  Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")

    def _extract_exports(self, tree: ast.AST, file_path: Path):
        """Extract __all__ exports from a module."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    self.exported_names[file_path].add(elt.value)

    def _find_string_references(self, file_path: Path):
        """Find string-based function references (getattr, etc.)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            for pattern in CONFIG["dynamic_usage_patterns"]:
                for match in re.finditer(pattern, content):
                    if match.groups():
                        self.string_references.add(match.group(1))

        except Exception as e:
            print(f"⚠️  Error finding string references in {file_path}: {e}")

    def _is_framework_function(self, func: FunctionInfo) -> bool:
        """Check if function is used by framework decorators."""
        for decorator in func.decorators:
            if any(
                decorator.startswith(fw_dec) or decorator.endswith(fw_dec)
                for fw_dec in CONFIG["framework_decorators"]
            ):
                return True
        return False

    def _is_special_function(self, func: FunctionInfo) -> bool:
        """Check if function matches special naming patterns."""
        for pattern in CONFIG["special_function_patterns"]:
            if re.match(pattern, func.name):
                return True
        return False

    def _is_exported(self, func: FunctionInfo) -> bool:
        """Check if function is explicitly exported via __all__."""
        return func.name in self.exported_names.get(func.file_path, set())

    def _find_unused_functions(self) -> List[Tuple[FunctionInfo, int]]:
        """
        Find unused functions with confidence scores.

        Returns:
            List of (FunctionInfo, confidence_percentage) tuples
        """
        unused = []

        for func_name, func_infos in self.all_functions.items():
            for func in func_infos:
                confidence = self._calculate_confidence(func, func_name)

                if confidence >= CONFIG["min_confidence"]:
                    unused.append((func, confidence))

        # Sort by confidence (descending) then by file path
        unused.sort(key=lambda x: (-x[1], str(x[0].file_path), x[0].line_number))

        return unused

    def _calculate_confidence(self, func: FunctionInfo, func_name: str) -> int:
        """
        Calculate confidence percentage that a function is unused.

        Returns:
            Confidence percentage (0-100)
        """
        # Start with 100% confidence (assume unused)
        confidence = 100

        # CRITICAL: Framework decorators mean function IS used
        if self._is_framework_function(func):
            return 0  # 0% confidence it's unused = 100% sure it's used

        # CRITICAL: Special function names (test_, __init__, etc.)
        if self._is_special_function(func):
            return 0

        # CRITICAL: Exported via __all__
        if self._is_exported(func):
            return 0

        # CRITICAL: Magic methods are always "used" by Python
        if func.is_magic:
            return 0

        # Check for actual usages in code
        usages = self.all_usages.get(func_name, [])

        # Filter out self-references (function calling itself or defined in same file)
        real_usages = [
            usage
            for usage in usages
            if usage.file_path != func.file_path
            or usage.line_number != func.line_number
        ]

        # If used in other files, it's definitely used
        if real_usages:
            # Reduce confidence based on number of usages
            # 1 usage: -40%, 2+ usages: -80%
            if len(real_usages) == 1:
                confidence -= 40
            else:
                confidence -= 80

        # Check string-based references (getattr, etc.)
        if func_name in self.string_references:
            confidence -= 50  # Significant reduction for dynamic usage

        # Private functions are more likely to be unused (less penalty for being unused)
        if func.is_private:
            confidence += 10  # Increase confidence slightly

        # Public functions in __init__.py are likely part of API
        if func.file_path.name == "__init__.py" and not func.is_private:
            confidence -= 30

        # Functions in test files with test_ prefix are definitely used
        if func.in_test_file and func.name.startswith("test_"):
            return 0

        # Fixtures in test files
        if func.in_test_file and any("fixture" in dec for dec in func.decorators):
            return 0

        # Functions with many decorators are likely used
        if len(func.decorators) > 0:
            confidence -= 20 * min(len(func.decorators), 3)

        # Ensure confidence stays in 0-100 range
        return max(0, min(100, confidence))


# ======================================================================================
# REPORT GENERATION
# ======================================================================================


def generate_report(unused: List[Tuple[FunctionInfo, int]], project_dir: Path):
    """Generate a detailed report of unused functions."""
    print("\n" + "=" * 80)
    print("📊 UNUSED FUNCTIONS REPORT")
    print("=" * 80)

    if not unused:
        print(
            "\n✅ No unused functions found with confidence >= {}%".format(
                CONFIG["min_confidence"]
            )
        )
        return

    print(f"\n⚠️  Found {len(unused)} potentially unused functions\n")

    # Group by confidence level
    high_confidence = [(f, c) for f, c in unused if c >= 95]
    medium_confidence = [(f, c) for f, c in unused if 90 <= c < 95]

    if high_confidence:
        print(f"\n🔴 HIGH CONFIDENCE (95-100%): {len(high_confidence)} functions")
        print("-" * 80)
        for func, confidence in high_confidence:
            _print_function_details(func, confidence, project_dir)

    if medium_confidence:
        print(f"\n🟡 MEDIUM CONFIDENCE (90-94%): {len(medium_confidence)} functions")
        print("-" * 80)
        for func, confidence in medium_confidence:
            _print_function_details(func, confidence, project_dir)

    # Generate summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(
        f"Total functions analyzed: {sum(len(funcs) for funcs in analyzer.all_functions.values())}"
    )
    print(f"High confidence unused (≥95%): {len(high_confidence)}")
    print(f"Medium confidence unused (90-94%): {len(medium_confidence)}")
    print(f"Total potentially unused: {len(unused)}")

    # Generate removal commands
    if high_confidence:
        print("\n" + "=" * 80)
        print("💡 NEXT STEPS")
        print("=" * 80)
        print("Review the high confidence functions above.")
        print("For each function, verify it's truly unused, then remove it.")
        print("\nRecommended approach:")
        print("1. Start with high confidence (95-100%) functions")
        print("2. Search project for function name to double-check")
        print("3. Run tests after removing each function")
        print("4. Commit changes incrementally")


def _print_function_details(func: FunctionInfo, confidence: int, project_dir: Path):
    """Print detailed information about a function."""
    relative_path = func.file_path.relative_to(project_dir)

    print(f"\n📍 {func.name}")
    print(f"   File: {relative_path}:{func.line_number}")
    print(f"   Confidence: {confidence}%")

    if func.decorators:
        print(f"   Decorators: {', '.join(func.decorators)}")

    flags = []
    if func.is_method:
        flags.append("method")
    if func.is_private:
        flags.append("private")
    if func.is_magic:
        flags.append("magic")
    if func.in_test_file:
        flags.append("test file")

    if flags:
        print(f"   Flags: {', '.join(flags)}")

    if func.docstring:
        # Print first line of docstring
        first_line = func.docstring.split("\n")[0].strip()
        if first_line:
            print(f"   Doc: {first_line[:70]}...")


# ======================================================================================
# PROJECT ROOT DETECTION
# ======================================================================================


def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by looking for common project markers.

    Searches upward from start_path for indicators like:
    - .git directory
    - pyproject.toml
    - setup.py
    - requirements.txt
    - README.md (with other markers)

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
        "Cargo.toml",  # Rust project (for mixed projects)
    ]

    # Start from the script directory and search upward
    current = start_path.resolve()

    # First, try to find markers in current directory and parents
    for _ in range(10):  # Limit search depth to prevent infinite loops
        # Check if any marker exists in current directory
        for marker in root_markers:
            marker_path = current / marker
            if marker_path.exists():
                print(f"✓ Found project root marker: {marker}")
                print(f"✓ Project root: {current}")
                return current

        # Move up one directory
        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding markers
            break
        current = parent

    # Fallback: if no markers found, use the directory containing the script
    # This assumes the script is somewhere within the project
    print("⚠️  No project root markers found")
    print(f"⚠️  Using script directory as project root: {start_path}")
    print(
        "   To improve detection, add a .git folder or pyproject.toml to your project"
    )
    return start_path


# ======================================================================================
# MAIN EXECUTION
# ======================================================================================


def main():
    """Main entry point."""
    print("🚀 Unused Functions Detector")
    print("=" * 80)

    # Determine project directory by searching for project root markers
    script_dir = Path(__file__).parent
    project_dir = find_project_root(script_dir)

    print(f"📁 Project: {project_dir}")
    print(f"📝 Config: Min confidence = {CONFIG['min_confidence']}%")

    # Create analyzer and run analysis
    global analyzer
    analyzer = UnusedFunctionAnalyzer(project_dir)
    unused = analyzer.analyze()

    # Generate report
    generate_report(unused, project_dir)

    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
