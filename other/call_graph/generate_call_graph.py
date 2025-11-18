"""
Interactive Call Graph Generator for Python Projects

This script generates interactive HTML visualizations of function/method calls
in your Python project using pyan3.

Usage:
    python generate_call_graph.py

Requirements:
    pip install pyan3

Output:
    - call_graph_{package}.html: Interactive HTML call graph visualization for each package
    - call_graph_{package}.dot: DOT file (if Graphviz not available)

Note:
    This script automatically detects the project root by searching for markers
    like .git, pyproject.toml, etc. It can be placed anywhere in the project.
"""

import subprocess
import sys
from pathlib import Path


# ======================================================================================
# PROJECT ROOT DETECTION
# ============================================================= =========================


def find_project_root(start_path: Path) -> Path:
    """
    Find the project root by looking for common project markers.

    Searches upward from start_path for indicators like:
    - .git directory
    - pyproject.toml
    - setup.py
    - requirements.txt
    - package.json

    Args:
        start_path: Path to start searching from (usually script location)

    Returns:
        Path to project root
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

    # Search upward through parent directories
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
    print("⚠️  No project root markers found")
    print(f"⚠️  Using script directory as project root: {start_path}")
    print(
        "   To improve detection, add a .git folder or pyproject.toml to your project"
    )
    return start_path


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    try:
        import pyan

        print("✓ pyan3 is available")
        return True
    except ImportError:
        print("✗ pyan3 is missing")
        print("\n⚠️  Missing dependencies detected!")
        print("Please install them manually:")
        print("  pip install pyan3")
        return False


def generate_call_graph(
    package_name="my_agent", output_file="call_graph.html", project_root=None
):
    """
    Generate interactive HTML call graph using pyan3.

    Args:
        package_name: Name of the package to analyze (default: my_agent)
        output_file: Output HTML file name
        project_root: Project root directory (for relative paths)
    """
    print(f"\nGenerating call graph for package '{package_name}'...")

    try:
        # Find all Python files in the package
        package_dir = (
            project_root / package_name if project_root else Path(package_name)
        )

        if not package_dir.exists():
            print(f"✗ Package directory '{package_dir}' not found.")
            return False

        # Collect all Python files in the package
        python_files = list(package_dir.rglob("*.py"))

        if not python_files:
            print(f"✗ No Python files found in package '{package_name}'.")
            return False

        print(f"  Found {len(python_files)} Python files")

        # Use pyan3 Python API directly to avoid command-line bugs
        try:
            from pyan.analyzer import CallGraphVisitor
            from pyan.visgraph import VisualGraph
            from pyan.writers import DotWriter
            import logging
            import threading

            print("  Analyzing code structure (this may take a moment)...")

            # Use threading for timeout on Windows
            dot_content = None
            error_msg = None

            def analyze_with_pyan():
                nonlocal dot_content, error_msg
                try:
                    # Create visitor
                    visitor = CallGraphVisitor(
                        [str(f) for f in python_files], logging.getLogger()
                    )

                    # Create visual graph
                    graph = VisualGraph.from_visitor(
                        visitor,
                        options={
                            "grouped": True,
                            "nested_groups": True,
                            "colored": True,
                            "annotated": True,
                        },
                    )

                    # Generate DOT content
                    writer = DotWriter(
                        graph,
                        options=["rankdir=LR"],
                        output=None,
                        logger=logging.getLogger(),
                    )

                    dot_content = writer.run()
                except Exception as e:
                    error_msg = str(e)

            # Run analysis in a thread with timeout
            analysis_thread = threading.Thread(target=analyze_with_pyan, daemon=True)
            analysis_thread.start()
            analysis_thread.join(timeout=30)  # 30 second timeout

            if analysis_thread.is_alive():
                print(
                    f"  ⚠️  Analysis timed out after 30 seconds, using simple import graph..."
                )
                dot_content = create_simple_import_graph(python_files, package_name)
            elif error_msg:
                print(
                    f"  ⚠️  pyan3 API failed ({error_msg}), using simple import graph..."
                )
                dot_content = create_simple_import_graph(python_files, package_name)
            elif not dot_content:
                print(f"  ⚠️  No content generated, using simple import graph...")
                dot_content = create_simple_import_graph(python_files, package_name)

        except Exception as e:
            print(f"  ⚠️  pyan3 setup failed ({e}), using simple import graph...")
            # Fallback: create a simple import graph
            dot_content = create_simple_import_graph(python_files, package_name)

        if not dot_content:
            print(f"✗ Failed to generate graph content")
            return False

        # Save DOT file
        dot_file = Path(output_file).with_suffix(".dot")
        with open(dot_file, "w") as f:
            f.write(dot_content)

        print(f"  Generated DOT file: {dot_file.name}")

        # Convert DOT to SVG using graphviz
        svg_file = Path(output_file).with_suffix(".svg")
        try:
            subprocess.run(
                ["dot", "-Tsvg", str(dot_file), "-o", str(svg_file)],
                check=True,
                capture_output=True,
            )
            print(f"  Generated SVG: {svg_file.name}")

            # Create interactive HTML wrapper
            create_html_wrapper(svg_file, output_file, package_name)

            # Clean up intermediate files
            dot_file.unlink()

            return True

        except FileNotFoundError:
            print("  ⚠️  Graphviz not found. DOT file saved.")
            print(f"  📁 Visualize at: https://dreampuf.github.io/GraphvizOnline/")
            print(f"     Or install Graphviz: https://graphviz.org/download/")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error converting to SVG: {e}")
            return False

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_simple_import_graph(python_files, package_name):
    """Create a simple DOT graph showing imports between modules."""
    import re

    graph_lines = [
        "digraph G {",
        "    rankdir=LR;",
        "    node [shape=box, style=filled, fillcolor=lightblue];",
        "",
    ]

    # Parse each file for imports
    imports = {}
    for file in python_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

            module_name = str(
                file.relative_to(file.parents[len(package_name.split("/"))])
            )
            module_name = (
                module_name.replace("\\", ".").replace("/", ".").replace(".py", "")
            )

            # Find imports
            import_pattern = r"from\s+(\S+)\s+import|import\s+(\S+)"
            for match in re.finditer(import_pattern, content):
                imported = match.group(1) or match.group(2)
                if imported and package_name in imported:
                    if module_name not in imports:
                        imports[module_name] = []
                    imports[module_name].append(imported)
        except:
            continue

    # Add nodes and edges
    for module, imported_modules in imports.items():
        safe_module = module.replace(".", "_")
        for imp in set(imported_modules):
            safe_imp = imp.replace(".", "_")
            graph_lines.append(f'    "{safe_module}" -> "{safe_imp}";')

    graph_lines.append("}")
    return "\n".join(graph_lines)


def create_html_wrapper(svg_file, html_file, package_name):
    """Create an interactive HTML wrapper for the SVG."""
    with open(svg_file, "r", encoding="utf-8") as f:
        svg_content = f.read()

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Call Graph - {package_name}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 100%;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        .controls {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        button {{
            padding: 8px 16px;
            margin-right: 10px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #0056b3;
        }}
        .svg-container {{
            overflow: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: grab;
        }}
        .svg-container:active {{
            cursor: grabbing;
        }}
        svg {{
            max-width: 100%;
            height: auto;
        }}
        .info {{
            margin-top: 15px;
            padding: 10px;
            background: #e7f3ff;
            border-left: 4px solid #007bff;
            border-radius: 4px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Call Graph: {package_name}</h1>
        <div class="controls">
            <button onclick="zoomIn()">🔍 Zoom In</button>
            <button onclick="zoomOut()">🔍 Zoom Out</button>
            <button onclick="resetZoom()">↺ Reset</button>
            <button onclick="fitToScreen()">⛶ Fit to Screen</button>
        </div>
        <div class="svg-container" id="svgContainer">
            {svg_content}
        </div>
        <div class="info">
            💡 <strong>Tip:</strong> Click and drag to pan, use the buttons above to zoom
        </div>
    </div>
    <script>
        let scale = 1;
        const svg = document.querySelector('svg');
        const container = document.getElementById('svgContainer');
        
        function zoomIn() {{
            scale *= 1.2;
            updateZoom();
        }}
        
        function zoomOut() {{
            scale /= 1.2;
            updateZoom();
        }}
        
        function resetZoom() {{
            scale = 1;
            updateZoom();
        }}
        
        function fitToScreen() {{
            const containerWidth = container.clientWidth;
            const svgWidth = svg.getBoundingClientRect().width / scale;
            scale = (containerWidth / svgWidth) * 0.9;
            updateZoom();
        }}
        
        function updateZoom() {{
            svg.style.transform = `scale(${{scale}})`;
            svg.style.transformOrigin = 'top left';
        }}
        
        // Make SVG draggable
        let isDragging = false;
        let startX, startY, scrollLeft, scrollTop;
        
        container.addEventListener('mousedown', (e) => {{
            isDragging = true;
            startX = e.pageX - container.offsetLeft;
            startY = e.pageY - container.offsetTop;
            scrollLeft = container.scrollLeft;
            scrollTop = container.scrollTop;
        }});
        
        container.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            e.preventDefault();
            const x = e.pageX - container.offsetLeft;
            const y = e.pageY - container.offsetTop;
            container.scrollLeft = scrollLeft - (x - startX);
            container.scrollTop = scrollTop - (y - startY);
        }});
        
        container.addEventListener('mouseup', () => {{
            isDragging = false;
        }});
        
        container.addEventListener('mouseleave', () => {{
            isDragging = false;
        }});
        
        // Mouse wheel zoom
        container.addEventListener('wheel', (e) => {{
            e.preventDefault();
            if (e.deltaY < 0) {{
                zoomIn();
            }} else {{
                zoomOut();
            }}
        }});
    </script>
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✓ Interactive HTML saved to {Path(html_file).name}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Interactive Call Graph Generator")
    print("=" * 60)

    # Step 0: Find project root and script directory
    script_dir = Path(__file__).parent
    project_root = find_project_root(script_dir)

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot proceed without required dependencies.")
        sys.exit(1)

    # Change to project root directory for analysis
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(project_root)

        # Step 2: Generate call graph data
        packages_to_analyze = ["my_agent", "api", "checkpointer"]

        print(f"\n📦 Analyzing packages: {', '.join(packages_to_analyze)}")

        success_count = 0
        for package in packages_to_analyze:
            # Use relative path from project root
            package_path = project_root / package
            # Output files go to the script's directory
            output_file = script_dir / f"call_graph_{package}.html"

            if not package_path.exists():
                print(f"\n⚠ Package '{package}' not found, skipping...")
                continue

            success = generate_call_graph(
                package, str(output_file), project_root=project_root
            )

            if success:
                success_count += 1
                print(f"  📁 Location: {output_file}")

        print("\n" + "=" * 60)
        if success_count > 0:
            print(f"✓ Done! Generated {success_count} call graph(s)")
            print(f"  📂 Output directory: {script_dir}")
            print("  🌐 Open the HTML files in your browser to explore!")
        else:
            print("❌ No call graphs were generated")
        print("=" * 60)

    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
