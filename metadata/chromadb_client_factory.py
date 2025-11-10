"""ChromaDB Client Factory

This module provides a factory for creating ChromaDB clients that can switch
between local PersistentClient and CloudClient based on environment configuration.

Configuration:
- CHROMA_USE_CLOUD: Set to 'true' to use cloud, 'false' or unset to use local
- CHROMA_API_KEY: API key for Chroma Cloud
- CHROMA_API_TENANT: Tenant ID for Chroma Cloud
- CHROMA_API_DATABASE: Database name for Chroma Cloud

Usage:
    from metadata.chromadb_client_factory import get_chromadb_client

    # Get client (automatically chooses cloud or local based on env)
    client = get_chromadb_client(local_path="/path/to/local/chromadb")

    # Get collection
    collection = client.get_collection(name="my_collection")
"""

import os
import sys
from pathlib import Path
from typing import Union
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path for imports
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(os.getcwd())

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import debug function
from api.utils.debug import print__chromadb_debug


def should_use_cloud() -> bool:
    """Determine if cloud client should be used based on environment variable.

    Returns:
        bool: True if CHROMA_USE_CLOUD is set to 'true', False otherwise
    """
    use_cloud_raw = os.getenv("CHROMA_USE_CLOUD", "false")
    use_cloud = use_cloud_raw.lower().strip()
    result = use_cloud in ("true", "1", "yes")

    print__chromadb_debug(f"🔍 [Factory] should_use_cloud() called")
    print__chromadb_debug(f"   CHROMA_USE_CLOUD raw value: '{use_cloud_raw}'")
    print__chromadb_debug(f"   Normalized value: '{use_cloud}'")
    print__chromadb_debug(f"   Result: {result}")

    return result


def get_chromadb_client(
    local_path: Union[str, Path] = None, collection_name: str = None
) -> Union[chromadb.PersistentClient, chromadb.CloudClient]:
    """Get ChromaDB client based on environment configuration.

    This function returns either a CloudClient or PersistentClient based on
    the CHROMA_USE_CLOUD environment variable.

    Args:
        local_path: Path to local ChromaDB directory (used if cloud is disabled)
        collection_name: Optional collection name for logging purposes

    Returns:
        Union[chromadb.PersistentClient, chromadb.CloudClient]: Configured client

    Raises:
        ValueError: If cloud is enabled but credentials are missing
        FileNotFoundError: If local path doesn't exist when using local client
    """
    print__chromadb_debug(f"🔧 [Factory] get_chromadb_client() called")
    print__chromadb_debug(f"   collection_name: {collection_name}")

    use_cloud = should_use_cloud()

    if use_cloud:
        print__chromadb_debug(
            f"🌐 [Factory] Cloud mode enabled - connecting to Chroma Cloud"
        )

        # Cloud configuration
        api_key = os.getenv("CHROMA_API_KEY", "").strip("',\"")
        tenant = os.getenv("CHROMA_API_TENANT", "").strip("',\"")
        database = os.getenv("CHROMA_API_DATABASE", "").strip("',\"")

        print__chromadb_debug(f"   API Key present: {bool(api_key)}")
        print__chromadb_debug(f"   Tenant: {tenant}")
        print__chromadb_debug(f"   Database: {database}")

        if not api_key or not tenant or not database:
            error_msg = (
                "Cloud mode enabled but missing credentials. "
                "Please set CHROMA_API_KEY, CHROMA_API_TENANT, and CHROMA_API_DATABASE"
            )
            print__chromadb_debug(f"❌ [Factory] {error_msg}")
            raise ValueError(error_msg)

        print__chromadb_debug(f"✅ [Factory] Creating CloudClient...")
        try:
            client = chromadb.CloudClient(
                api_key=api_key, tenant=tenant, database=database
            )
            print__chromadb_debug(f"✅ [Factory] CloudClient created successfully")
            print__chromadb_debug(f"   Client type: {type(client).__name__}")
            if collection_name:
                print__chromadb_debug(f"   Target collection: {collection_name}")
            return client
        except Exception as exc:
            print__chromadb_debug(f"❌ [Factory] Failed to create CloudClient: {exc}")
            raise

    else:
        print__chromadb_debug(
            f"📂 [Factory] Local mode enabled - using PersistentClient"
        )
        print__chromadb_debug(f"   local_path: {local_path}")

        # Local configuration
        if local_path is None:
            error_msg = "local_path must be provided when using local ChromaDB"
            print__chromadb_debug(f"❌ [Factory] {error_msg}")
            raise ValueError(error_msg)

        local_path = Path(local_path)
        print__chromadb_debug(f"   Checking local path: {local_path}")
        print__chromadb_debug(f"   Path exists: {local_path.exists()}")
        print__chromadb_debug(
            f"   Is directory: {local_path.is_dir() if local_path.exists() else 'N/A'}"
        )

        if not local_path.exists():
            print__chromadb_debug(
                f"⚠️ [Factory] Local ChromaDB path not found, creating it: {local_path}"
            )
            local_path.mkdir(parents=True, exist_ok=True)
            print__chromadb_debug(f"✅ [Factory] Directory created: {local_path}")

        print__chromadb_debug(f"✅ [Factory] Creating PersistentClient...")
        try:
            client = chromadb.PersistentClient(path=str(local_path))
            print__chromadb_debug(f"✅ [Factory] PersistentClient created successfully")
            print__chromadb_debug(f"   Client type: {type(client).__name__}")
            print__chromadb_debug(f"   Path: {local_path}")
            if collection_name:
                print__chromadb_debug(f"   Target collection: {collection_name}")
            return client
        except Exception as exc:
            print__chromadb_debug(
                f"❌ [Factory] Failed to create PersistentClient: {exc}"
            )
            raise


def get_chromadb_collection(
    collection_name: str,
    local_path: Union[str, Path] = None,
) -> chromadb.Collection:
    """Get a ChromaDB collection using the appropriate client.

    Args:
        collection_name: Name of the collection to retrieve
        local_path: Path to local ChromaDB directory (used if cloud is disabled)

    Returns:
        chromadb.Collection: The requested collection

    Raises:
        ValueError: If cloud is enabled but credentials are missing
        FileNotFoundError: If local path doesn't exist when using local client
    """
    use_cloud = should_use_cloud()

    print__chromadb_debug(f"📦 [Factory] get_chromadb_collection() called")
    print__chromadb_debug(f"   collection_name: {collection_name}")
    print__chromadb_debug(f"   mode: {'☁️ Cloud' if use_cloud else '📂 Local'}")
    if not use_cloud:
        print__chromadb_debug(f"   local_path: {local_path}")

    client = get_chromadb_client(local_path=local_path, collection_name=collection_name)

    print__chromadb_debug(f"   Retrieving collection '{collection_name}'...")
    try:
        collection = client.get_collection(name=collection_name)
        print__chromadb_debug(
            f"✅ [Factory] Collection '{collection_name}' retrieved successfully"
        )
        print__chromadb_debug(f"   Collection type: {type(collection).__name__}")
        return collection
    except Exception as exc:
        print__chromadb_debug(
            f"❌ [Factory] Failed to retrieve collection '{collection_name}': {exc}"
        )
        raise


def get_or_create_chromadb_collection(
    collection_name: str, local_path: Union[str, Path] = None, metadata: dict = None
) -> chromadb.Collection:
    """Get or create a ChromaDB collection using the appropriate client.

    Args:
        collection_name: Name of the collection to get or create
        local_path: Path to local ChromaDB directory (used if cloud is disabled)
        metadata: Optional metadata for the collection (e.g., {"hnsw:space": "cosine"})

    Returns:
        chromadb.Collection: The requested or created collection

    Raises:
        ValueError: If cloud is enabled but credentials are missing
        FileNotFoundError: If local path doesn't exist when using local client
    """
    use_cloud = should_use_cloud()

    print__chromadb_debug(f"🔨 [Factory] get_or_create_chromadb_collection() called")
    print__chromadb_debug(f"   collection_name: {collection_name}")
    print__chromadb_debug(f"   mode: {'☁️ Cloud' if use_cloud else '📂 Local'}")
    if not use_cloud:
        print__chromadb_debug(f"   local_path: {local_path}")
    print__chromadb_debug(f"   metadata: {metadata}")

    client = get_chromadb_client(local_path=local_path, collection_name=collection_name)

    try:
        print__chromadb_debug(
            f"   Attempting to create collection '{collection_name}'..."
        )
        # Try to create the collection
        if metadata:
            collection = client.create_collection(
                name=collection_name, metadata=metadata
            )
            print__chromadb_debug(
                f"✅ [Factory] Created new collection '{collection_name}' with metadata"
            )
        else:
            collection = client.create_collection(name=collection_name)
            print__chromadb_debug(
                f"✅ [Factory] Created new collection '{collection_name}' (no metadata)"
            )
    except Exception as create_error:
        print__chromadb_debug(
            f"   Collection already exists, retrieving it... ({create_error})"
        )
        # Collection already exists, get it
        try:
            collection = client.get_collection(name=collection_name)
            print__chromadb_debug(
                f"✅ [Factory] Using existing collection '{collection_name}'"
            )
        except Exception as get_error:
            print__chromadb_debug(
                f"❌ [Factory] Failed to get existing collection '{collection_name}': {get_error}"
            )
            raise

    print__chromadb_debug(f"   Collection type: {type(collection).__name__}")
    return collection
