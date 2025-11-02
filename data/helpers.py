"""
Helper Functions for PDF Processing Scripts

This module provides shared helper functions for all pdf_to_chromadb__* scripts,
including file I/O operations with dynamic suffix handling based on the calling script name.

Functions:
- save_parsed_text_to_file: Save parsed text with automatic suffix from calling script
- load_parsed_text_from_file: Load parsed text from file
- extract_script_suffix: Extract parsing method suffix from script filename
"""

import os
import sys
from pathlib import Path
from typing import Optional


def extract_script_suffix(script_path: Optional[str] = None) -> str:
    """
    Extract the suffix from a pdf_to_chromadb__* script filename.

    The suffix is the part after the double underscore (__) in the script name.
    For example:
    - pdf_to_chromadb__llamaparse_MarkdownElementNodeParser.py -> llamaparse_MarkdownElementNodeParser
    - pdf_to_chromadb__azure_doc_intelligence.py -> azure_doc_intelligence
    - pdf_to_chromadb__pymupdf.py -> pymupdf

    Args:
        script_path: Path to the script file (optional, defaults to calling script)

    Returns:
        The extracted suffix, or 'unknown' if not found
    """
    if script_path is None:
        # Get the calling script's path from the call stack
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            script_path = caller_frame.f_globals.get("__file__", "")

    if not script_path:
        return "unknown"

    # Get just the filename without extension
    filename = Path(script_path).stem

    # Check if it follows the pdf_to_chromadb__* pattern
    if "__" in filename:
        # Extract everything after the double underscore
        suffix = filename.split("__", 1)[1]
        return suffix

    return "unknown"


def save_parsed_text_to_file(
    text: str,
    pdf_filename: str,
    output_dir: Optional[Path] = None,
    script_path: Optional[str] = None,
) -> Path:
    """
    Save parsed text to a file with automatic suffix from the calling script.

    The output filename format is: {pdf_basename}_{script_suffix}_parsed.txt
    For example, if called from pdf_to_chromadb__azure_doc_intelligence.py with PDF "report.pdf":
    - Output: report.pdf_azure_doc_intelligence_parsed.txt

    Args:
        text: The parsed text content to save
        pdf_filename: Original PDF filename (can be basename or full path)
        output_dir: Directory to save the file (optional, defaults to script directory)
        script_path: Path to the calling script (optional, auto-detected)

    Returns:
        Path object pointing to the saved file

    Raises:
        IOError: If file cannot be written
    """
    # Extract the suffix from the calling script
    suffix = extract_script_suffix(script_path)

    # Get PDF basename (without directory)
    pdf_basename = os.path.basename(pdf_filename)

    # Create output filename
    output_filename = f"{pdf_basename}_{suffix}_parsed.txt"

    # Determine output directory
    if output_dir is None:
        # Get the directory of the calling script
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            caller_file = caller_frame.f_globals.get("__file__", "")
            if caller_file:
                output_dir = Path(caller_file).parent
            else:
                output_dir = Path.cwd()
        else:
            output_dir = Path.cwd()

    output_path = output_dir / output_filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Parsed text saved to: {output_path}")
        print(f"📊 Content length: {len(text):,} characters")

        # Log to debug if available
        try:
            sys.path.insert(0, str(output_dir.parent))
            from api.utils.debug import print__chromadb_debug

            print__chromadb_debug(
                f"💾 Successfully saved parsed text to: {output_path}"
            )
        except ImportError:
            pass

        return output_path

    except Exception as e:
        print(f"❌ Error saving parsed text: {str(e)}")
        raise IOError(f"Failed to save parsed text to {output_path}: {str(e)}")


def load_parsed_text_from_file(
    pdf_filename: str,
    input_dir: Optional[Path] = None,
    script_path: Optional[str] = None,
) -> str:
    """
    Load parsed text from a file with automatic suffix matching.

    The expected filename format is: {pdf_basename}_{script_suffix}_parsed.txt
    For example, if called from pdf_to_chromadb__azure_doc_intelligence.py with PDF "report.pdf":
    - Looks for: report.pdf_azure_doc_intelligence_parsed.txt

    Args:
        pdf_filename: Original PDF filename (can be basename or full path)
        input_dir: Directory to load the file from (optional, defaults to script directory)
        script_path: Path to the calling script (optional, auto-detected)

    Returns:
        The loaded text content

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If file cannot be read
    """
    # Extract the suffix from the calling script
    suffix = extract_script_suffix(script_path)

    # Get PDF basename (without directory)
    pdf_basename = os.path.basename(pdf_filename)

    # Create expected filename
    input_filename = f"{pdf_basename}_{suffix}_parsed.txt"

    # Determine input directory
    if input_dir is None:
        # Get the directory of the calling script
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            caller_file = caller_frame.f_globals.get("__file__", "")
            if caller_file:
                input_dir = Path(caller_file).parent
            else:
                input_dir = Path.cwd()
        else:
            input_dir = Path.cwd()

    input_path = input_dir / input_filename

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"✅ Loaded parsed text from: {input_path}")
        print(f"📊 Content length: {len(text):,} characters")

        # Log to debug if available
        try:
            sys.path.insert(0, str(input_dir.parent))
            from api.utils.debug import print__chromadb_debug

            print__chromadb_debug(
                f"💾 Successfully loaded parsed text from: {input_path}"
            )
        except ImportError:
            pass

        return text

    except FileNotFoundError:
        error_msg = f"Parsed text file not found: {input_path}"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error loading parsed text from {input_path}: {str(e)}"
        print(f"❌ {error_msg}")
        raise IOError(error_msg)


# Legacy function names for backward compatibility
def save_parsed_text_to_file_legacy(text: str, file_path: str) -> None:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use save_parsed_text_to_file() instead for automatic suffix handling.

    Args:
        text: The parsed text content
        file_path: Full path where to save the text file
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Parsed text saved to: {file_path}")
        print(f"📊 Content length: {len(text):,} characters")
    except Exception as e:
        print(f"❌ Error saving parsed text: {str(e)}")
        raise


def load_parsed_text_from_file_legacy(file_path: str) -> str:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use load_parsed_text_from_file() instead for automatic suffix handling.

    Args:
        file_path: Path to the parsed text file

    Returns:
        The parsed text content
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ Loaded parsed text from: {file_path}")
        return text
    except FileNotFoundError:
        print(f"❌ Parsed text file not found: {file_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading parsed text: {str(e)}")
        raise
