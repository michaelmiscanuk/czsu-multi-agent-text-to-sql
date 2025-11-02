# Azure Document Intelligence Implementation Guide

## Overview
This guide explains how to complete the Azure Document Intelligence implementation for the czsu-multi-agent-text-to-sql project, replacing LlamaParse with Azure DI while maintaining all other functionality.

## Status Summary

### ✅ Completed
1. **helpers.py created** - File I/O functions with dynamic suffix extraction
2. **.env updated** - Added Azure DI credentials placeholders
3. **.env.example updated** - Complete example configuration

### 🔄 In Progress
4. **Azure DI parsing implementation** - Need to complete in pdf_to_chromadb__azure_doc_intelligence.py

### ⏳ Pending
5. **Update all pdf_to_chromadb scripts** - Replace file I/O calls with helpers.py functions
6. **Testing and validation** - Full pipeline testing

---

## 1. Azure Document Intelligence Implementation

### Key Changes Needed in `pdf_to_chromadb__azure_doc_intelligence.py`

#### A. Replace Imports Section (lines ~1-100)

Replace LlamaParse imports with:
```python
# Azure Document Intelligence imports
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        AnalyzeResult,
        ContentFormat,
        DocumentContentFormat
    )
    AZURE_DOC_INTELLIGENCE_AVAILABLE = True
except ImportError:
    print("Warning: Azure Document Intelligence SDK not available.")
    print("Install with: pip install azure-ai-documentintelligence")
    AZURE_DOC_INTELLIGENCE_AVAILABLE = False
```

#### B. Update Configuration Section (lines ~200-300)

Replace:
```python
LLAMAPARSE_API_KEY = os.environ.get("LLAMAPARSE_API_KEY", "")
PDF_PARSING_METHOD = "llamaparse"
```

With:
```python
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
PDF_PARSING_METHOD = "azure_doc_intelligence"
```

#### C. Replace Parsing Function (lines ~800-1200)

Replace `extract_text_with_llamaparse()` and `extract_text_with_llamaparse_async_monitoring()` with:

```python
def extract_text_with_azure_doc_intelligence(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using Azure Document Intelligence layout model.
    
    Features:
    - Uses prebuilt-layout model for complex tables and charts
    - Markdown output format (HTML tables in v4.0)
    - Multi-page table consolidation
    - Section and heading detection
    - Figure extraction with captions
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text and metadata for each page
    """
    if not AZURE_DOC_INTELLIGENCE_AVAILABLE:
        raise ValueError("Azure Document Intelligence SDK not available. Install with: pip install azure-ai-documentintelligence")
    
    if not AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or not AZURE_DOCUMENT_INTELLIGENCE_KEY:
        raise ValueError(
            "Azure Document Intelligence not configured. "
            "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY in .env file"
        )
    
    print__chromadb_debug(f"📄 Opening PDF with Azure Document Intelligence: {pdf_path}")
    
    try:
        pdf_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        print(f"\n🚀 Starting Azure Document Intelligence processing...")
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"📊 Size: {pdf_size:.1f} MB")
        
        # Initialize client
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)
        )
        
        # Start analysis
        print(f"📤 Uploading document...")
        start_time = time.time()
        
        with open(pdf_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=f,
                content_type="application/pdf",
                output_content_format=ContentFormat.MARKDOWN,
            )
        
        print(f"⏳ Analyzing document...")
        result: AnalyzeResult = poller.result()
        
        elapsed_time = time.time() - start_time
        print(f"✅ Analysis completed in {elapsed_time:.1f} seconds!")
        
        # Extract markdown content
        if not result.content:
            raise ValueError("No content extracted from document")
        
        markdown_content = result.content
        print(f"📊 Extracted {len(markdown_content):,} characters")
        
        # Process into pages
        pages_data = []
        
        # Split by page markers if present
        page_pattern = r"---\s*Page\s*(\d+)\s*---"
        page_splits = re.split(page_pattern, markdown_content)
        
        if len(page_splits) > 1:
            # Has page markers
            for i in range(1, len(page_splits), 2):
                if i < len(page_splits):
                    page_num = int(page_splits[i])
                    page_text = page_splits[i + 1].strip() if i + 1 < len(page_splits) else ""
                    
                    if page_text:
                        page_info = {
                            "text": page_text,
                            "page_number": page_num,
                            "char_count": len(page_text),
                            "word_count": len(page_text.split()),
                            "source_file": os.path.basename(pdf_path),
                            "parsing_method": "azure_doc_intelligence",
                        }
                        pages_data.append(page_info)
                        print(f"   ✅ Page {page_num}: {len(page_text):,} chars")
        else:
            # No page markers - single document
            page_info = {
                "text": markdown_content,
                "page_number": 1,
                "char_count": len(markdown_content),
                "word_count": len(markdown_content.split()),
                "source_file": os.path.basename(pdf_path),
                "parsing_method": "azure_doc_intelligence",
            }
            pages_data.append(page_info)
        
        # Save combined text using helpers
        combined_text_parts = [p["text"] for p in pages_data]
        if combined_text_parts:
            # Import here to use dynamic suffix
            from data.helpers import save_parsed_text_to_file
            
            # Add page separators for consistency
            combined_text = ""
            for p in pages_data:
                if combined_text:
                    combined_text += f"\n\n--- Page {p['page_number']} ---\n\n"
                combined_text += p['text']
            
            save_parsed_text_to_file(
                text=combined_text,
                pdf_filename=os.path.basename(pdf_path)
            )
        
        # Summary
        total_chars = sum(p["char_count"] for p in pages_data)
        total_words = sum(p["word_count"] for p in pages_data)
        
        print(f"\n🎉 Azure DI Processing Complete!")
        print(f"   📄 Pages: {len(pages_data)}")
        print(f"   📝 Characters: {total_chars:,}")
        print(f"   🔤 Words: {total_words:,}")
        print(f"   ⏱️  Time: {elapsed_time:.1f}s")
        print(f"   📊 Speed: {total_chars/elapsed_time:.0f} chars/sec")
        
        print__chromadb_debug(f"Successfully extracted {len(pages_data)} pages with Azure DI")
        return pages_data
        
    except Exception as e:
        print__chromadb_debug(f"Error with Azure DI: {str(e)}")
        print(f"\n❌ Azure DI failed: {str(e)}")
        raise
```

#### D. Update Main Execution (lines ~2800-2900)

Replace parsing section:
```python
if PARSE_WITH_AZURE_DOC_INTELLIGENCE:
    print("\n🔍 OPERATION 1: PARSING WITH AZURE DOCUMENT INTELLIGENCE")
    print("-" * 80)
    
    for pdf_filename in PDF_FILENAMES:
        pdf_path = SCRIPT_DIR / pdf_filename
        
        if not pdf_path.exists():
            print(f"❌ PDF not found: {pdf_path}")
            continue
        
        try:
            print(f"\n📄 Processing: {pdf_filename}")
            pages_data = extract_text_with_azure_doc_intelligence(str(pdf_path))
            print(f"✅ Successfully parsed {pdf_filename}")
        except Exception as e:
            print(f"❌ Failed to parse {pdf_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
```

---

## 2. Update All pdf_to_chromadb Scripts

### Scripts to Update:
- pdf_to_chromadb__llamaparse_MarkdownElementNodeParser.py
- Any other pdf_to_chromadb__*.py scripts

### Changes Needed:

#### A. Add Import at Top (after other imports)
```python
from data.helpers import save_parsed_text_to_file, load_parsed_text_from_file
```

#### B. Remove Old save/load Functions
Delete these functions:
- `save_parsed_text_to_file()` - OLD VERSION
- `load_parsed_text_from_file()` - OLD VERSION

#### C. Update Save Calls

**OLD CODE:**
```python
# Generate parsed text filename dynamically
parsed_text_filename = f"{os.path.basename(pdf_path)}_llamaparse_parsed.txt"
parsed_text_path = SCRIPT_DIR / parsed_text_filename
save_parsed_text_to_file(combined_text, str(parsed_text_path))
```

**NEW CODE:**
```python
# Save with automatic suffix from script name
save_parsed_text_to_file(
    text=combined_text,
    pdf_filename=os.path.basename(pdf_path)
)
```

#### D. Update Load Calls

**OLD CODE:**
```python
PARSED_TEXT_FILENAME = f"{PDF_FILENAME}_llamaparse_parsed.txt"
PARSED_TEXT_PATH = SCRIPT_DIR / PARSED_TEXT_FILENAME
parsed_text = load_parsed_text_from_file(str(PARSED_TEXT_PATH))
```

**NEW CODE:**
```python
parsed_text = load_parsed_text_from_file(
    pdf_filename=PDF_FILENAME
)
```

---

## 3. Installation Requirements

### Install Azure Document Intelligence SDK:
```bash
pip install azure-ai-documentintelligence
```

### Or add to pyproject.toml:
```toml
dependencies = [
    ...
    "azure-ai-documentintelligence>=1.0.0",
    ...
]
```

Then run:
```bash
uv sync
```

---

## 4. Configuration

### Update .env File with Real Values:
```properties
# Azure Document Intelligence (get from Azure Portal)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource-name.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-actual-key-here
```

### To Create Azure DI Resource:
1. Go to Azure Portal
2. Create new "Document Intelligence" resource
3. Copy endpoint and key from "Keys and Endpoint" section
4. Paste into .env file

---

## 5. Testing Procedure

### Test 1: Parse Single PDF
```python
PARSE_WITH_AZURE_DOC_INTELLIGENCE = 1
CHUNK_AND_STORE = 0
DO_TESTING = 0
PDF_FILENAMES = ["96_97_141.pdf"]  # Small test file

# Run script
python data/pdf_to_chromadb__azure_doc_intelligence.py
```

**Expected Output:**
- PDF analysis completes successfully
- Markdown content extracted
- File saved as: `96_97_141.pdf_azure_doc_intelligence_parsed.txt`

### Test 2: Chunk and Store
```python
PARSE_WITH_AZURE_DOC_INTELLIGENCE = 0
CHUNK_AND_STORE = 1
DO_TESTING = 0

# Run script
python data/pdf_to_chromadb__azure_doc_intelligence.py
```

**Expected Output:**
- Parsed text loaded successfully
- Chunks created with quality metrics
- ChromaDB storage completes
- No errors about missing files

### Test 3: Search
```python
PARSE_WITH_AZURE_DOC_INTELLIGENCE = 0
CHUNK_AND_STORE = 0
DO_TESTING = 1
TEST_QUERY = "Jaky je prutok reky Metuje?"

# Run script
python data/pdf_to_chromadb__azure_doc_intelligence.py
```

**Expected Output:**
- Search returns relevant results
- Hybrid search combines semantic + BM25
- Cohere reranking improves results

---

## 6. Comparison with LlamaParse

### Azure DI Advantages:
✅ Native Microsoft Azure integration
✅ No third-party API dependency
✅ Potentially lower latency (same region)
✅ Enterprise-grade SLA and support
✅ Built-in compliance (GDPR, HIPAA, etc.)
✅ Markdown output with HTML tables (v4.0)
✅ Multi-page table consolidation
✅ Figure extraction with captions

### LlamaParse Advantages:
✅ Specialized for LLM workflows
✅ Custom parsing instructions
✅ Vision model for chart reading
✅ Continuous mode for long tables

### Output Format Comparison:

**LlamaParse:**
- Pipe-delimited markdown tables: `| Col1 | Col2 |`
- Custom page separators
- Vision-based chart extraction

**Azure DI v4.0:**
- HTML tables in markdown: `<table><tr><td>Cell</td></tr></table>`
- Standard page markers
- Figure objects with bounding regions

**Impact on Chunking:**
- MarkdownElementNodeParser handles both formats
- HTML tables preserve structure better for complex cases
- Both work with hybrid search

---

## 7. Troubleshooting

### Error: "Azure Document Intelligence SDK not available"
**Solution:** Install package:
```bash
pip install azure-ai-documentintelligence
```

### Error: "Azure Document Intelligence not configured"
**Solution:** Set environment variables in .env:
```properties
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://...
AZURE_DOCUMENT_INTELLIGENCE_KEY=...
```

### Error: "No content extracted from document"
**Solution:** Check PDF file:
- Is it text-based or scanned image?
- Is file size under 500MB?
- Try with a simpler PDF first

### Error: "Module not found: data.helpers"
**Solution:** Ensure helpers.py exists in data/ folder and Python path is correct

### Poor Search Results
**Solution:** Check chunking quality:
```python
# Look for quality metrics in output:
# - Quality score should be > 70/100
# - Chunks with context > 50%
# - Chunks with numbers > 30% (for statistical docs)
```

---

## 8. Next Steps

After completing implementation:

1. ✅ **Test with sample PDF** - Verify parsing works
2. ✅ **Compare outputs** - Azure DI vs LlamaParse side-by-side
3. ✅ **Evaluate chunk quality** - Check table preservation
4. ✅ **Test search accuracy** - Compare retrieval results
5. ✅ **Performance benchmark** - Measure speed and cost
6. ✅ **Update documentation** - Document findings

---

## Summary

The implementation replaces LlamaParse with Azure Document Intelligence while:
- ✅ Maintaining same architecture
- ✅ Using helpers.py for file I/O
- ✅ Preserving chunking strategy
- ✅ Keeping hybrid search unchanged
- ✅ Supporting all existing features

Key files modified:
1. data/helpers.py - NEW (file I/O with dynamic suffix)
2. data/pdf_to_chromadb__azure_doc_intelligence.py - UPDATED (Azure DI parsing)
3. .env - UPDATED (Azure DI credentials)
4. .env.example - UPDATED (example config)

All pdf_to_chromadb__* scripts should be updated to use helpers.py functions.
