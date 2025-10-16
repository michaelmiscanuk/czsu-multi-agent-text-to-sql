# ChromaDB Cloud/Local Switching

This document explains the ChromaDB cloud/local switching functionality that has been added to the project.

## Overview

The project now supports seamlessly switching between local ChromaDB (PersistentClient) and Chroma Cloud (CloudClient) using a simple environment variable configuration.

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Chroma Cloud Configuration
CHROMA_USE_CLOUD=true                           # Set to 'true' to use cloud, 'false' for local
CHROMA_API_KEY=your-api-key-here
CHROMA_API_TENANT=your-tenant-id-here
CHROMA_API_DATABASE=your-database-name-here
```

### Switching Between Cloud and Local

**To use Chroma Cloud:**
```env
CHROMA_USE_CLOUD=true
```

**To use Local ChromaDB:**
```env
CHROMA_USE_CLOUD=false
```

## Usage

### In Your Code

The factory automatically handles cloud/local switching:

```python
from metadata.chromadb_client_factory import get_chromadb_client, get_chromadb_collection

# Get a client (automatically chooses cloud or local)
client = get_chromadb_client(
    local_path="/path/to/local/chromadb",  # Only used if CHROMA_USE_CLOUD=false
    collection_name="my_collection"
)

# Get a collection directly
collection = get_chromadb_collection(
    collection_name="my_collection",
    local_path="/path/to/local/chromadb"  # Only used if CHROMA_USE_CLOUD=false
)

# Use the collection normally
results = collection.query(
    query_embeddings=[embedding],
    n_results=10
)
```

## Migration Script

To copy your local ChromaDB collections to Chroma Cloud:

```bash
python chromadb_local_to_cloud.py
```

This script will:
1. Read from local ChromaDB directories specified in `LOCAL_CHROMADB_PATHS`
2. Copy all collections to Chroma Cloud
3. Overwrite existing collections (no duplicates)
4. Preserve all documents, embeddings, and metadata

### Configuration

Edit `LOCAL_CHROMADB_PATHS` in `chromadb_local_to_cloud.py`:

```python
LOCAL_CHROMADB_PATHS = [
    BASE_DIR / "metadata" / "czsu_chromadb",
    BASE_DIR / "data" / "pdf_chromadb_llamaparse",
    # Add more paths as needed
]
```

## Testing

Test the cloud/local switching functionality:

```bash
python test_chromadb_switching.py
```

This will:
1. Check your environment configuration
2. Test client creation
3. Test collection access
4. Display a summary of results

## Files Modified

### New Files
- `metadata/chromadb_client_factory.py` - Factory for creating ChromaDB clients
- `chromadb_local_to_cloud.py` - Script to migrate local ChromaDB to cloud
- `test_chromadb_switching.py` - Test script for cloud/local switching
- `CHROMADB_SWITCHING.md` - This documentation file

### Modified Files
- `my_agent/utils/nodes.py` - Updated to use factory for ChromaDB client creation
- `metadata/04_create_and_load_chromadb.py` - Updated to use factory
- `data/pdf_to_chromadb.py` - Updated to use factory

## Benefits

1. **Flexibility**: Easy switching between local development and cloud production
2. **No Code Changes**: Switch by changing environment variable only
3. **Backward Compatible**: Existing code continues to work
4. **Consistent API**: Same interface for both cloud and local
5. **Easy Migration**: Script provided to copy local data to cloud

## Troubleshooting

### Cloud Connection Issues

If you get cloud connection errors:
1. Check that `CHROMA_API_KEY`, `CHROMA_API_TENANT`, and `CHROMA_API_DATABASE` are set correctly
2. Verify credentials at https://www.trychroma.com/
3. Check network connectivity

### Local Path Issues

If you get local path errors:
1. Verify the local ChromaDB directory exists
2. Check that collections have been created locally
3. Ensure proper file permissions

### Testing

Run the test script to diagnose issues:
```bash
python test_chromadb_switching.py
```

## Examples

### Example 1: Development (Local)

```env
CHROMA_USE_CLOUD=false
```

Your code will use local ChromaDB at the specified paths.

### Example 2: Production (Cloud)

```env
CHROMA_USE_CLOUD=true
CHROMA_API_KEY=ck-5AUDXwgSNjAkNUTe6tRsh9k2crCgbb6FVNGhSZWVqjLg
CHROMA_API_TENANT=f18152e6-d8fb-4745-9415-7a95930330d9
CHROMA_API_DATABASE=CZSU-Multi-Agent-Text-to-SQL
```

Your code will use Chroma Cloud with the specified credentials.

### Example 3: Migration Workflow

1. Develop locally with `CHROMA_USE_CLOUD=false`
2. When ready, run `python chromadb_local_to_cloud.py` to copy data to cloud
3. Switch to `CHROMA_USE_CLOUD=true` for production
4. Test with `python test_chromadb_switching.py`

## Support

For issues or questions:
1. Check this documentation
2. Run the test script: `python test_chromadb_switching.py`
3. Review error messages and logs
4. Check Chroma documentation: https://docs.trychroma.com/
