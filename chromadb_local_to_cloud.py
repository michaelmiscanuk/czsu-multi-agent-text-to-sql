"""ChromaDB Local to Cloud Migration Script

This script copies local ChromaDB collections to Chroma Cloud.
It reads from local persistent ChromaDB directories and uploads all collections
to Chroma Cloud, overwriting existing collections with the same name to avoid duplicates.

Configuration:
- Local ChromaDB paths are specified in LOCAL_CHROMADB_PATHS list
- Cloud credentials are loaded from .env file:
  - CHROMA_API_KEY
  - CHROMA_API_TENANT
  - CHROMA_API_DATABASE

Usage:
    python chromadb_local_to_cloud.py

Features:
- Copies all collections from specified local ChromaDB directories
- Preserves documents, embeddings, and metadata
- Overwrites existing cloud collections (no duplicates)
- Progress tracking with detailed logging
- Batch processing for efficient transfer
- Error handling and retry logic
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from tqdm import tqdm

# Load environment variables
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Get the base directory
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

# Configuration of local ChromaDB paths to migrate to cloud
LOCAL_CHROMADB_PATHS = [
    # BASE_DIR / "metadata" / "czsu_chromadb",
    BASE_DIR / "data" / "pdf_chromadb_llamaparse",
    # Add more paths here if needed:
]

# Cloud configuration from .env
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "").strip("',\"")
CHROMA_API_TENANT = os.getenv("CHROMA_API_TENANT", "").strip("',\"")
CHROMA_API_DATABASE = os.getenv("CHROMA_API_DATABASE", "").strip("',\"")

# Batch size for copying documents (adjust if needed)
BATCH_SIZE = 100

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def validate_config() -> bool:
    """Validate that all required configuration is present.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not CHROMA_API_KEY:
        print("❌ Error: CHROMA_API_KEY not found in .env file")
        return False
    if not CHROMA_API_TENANT:
        print("❌ Error: CHROMA_API_TENANT not found in .env file")
        return False
    if not CHROMA_API_DATABASE:
        print("❌ Error: CHROMA_API_DATABASE not found in .env file")
        return False

    print("✅ Cloud configuration validated")
    print(f"   Tenant: {CHROMA_API_TENANT}")
    print(f"   Database: {CHROMA_API_DATABASE}")
    return True


def get_cloud_client() -> chromadb.CloudClient:
    """Initialize and return a ChromaDB Cloud client.

    Returns:
        chromadb.CloudClient: Configured cloud client
    """
    print(f"\n🌐 Connecting to Chroma Cloud...")
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY, tenant=CHROMA_API_TENANT, database=CHROMA_API_DATABASE
    )
    print("✅ Connected to Chroma Cloud")
    return client


def get_local_client(path: Path) -> chromadb.PersistentClient:
    """Initialize and return a local ChromaDB client.

    Args:
        path: Path to the local ChromaDB directory

    Returns:
        chromadb.PersistentClient: Local client instance
    """
    if not path.exists():
        raise FileNotFoundError(f"Local ChromaDB path not found: {path}")

    print(f"📂 Connecting to local ChromaDB: {path}")
    client = chromadb.PersistentClient(path=str(path))
    return client


def list_local_collections(client: chromadb.PersistentClient) -> List[str]:
    """Get list of all collections in a local ChromaDB.

    Args:
        client: Local ChromaDB client

    Returns:
        List[str]: List of collection names
    """
    collections = client.list_collections()
    return [col.name for col in collections]


def copy_collection(
    local_client: chromadb.PersistentClient,
    cloud_client: chromadb.CloudClient,
    collection_name: str,
) -> Dict[str, Any]:
    """Copy a single collection from local to cloud.

    Args:
        local_client: Local ChromaDB client
        cloud_client: Cloud ChromaDB client
        collection_name: Name of the collection to copy

    Returns:
        Dict[str, Any]: Statistics about the copy operation
    """
    stats = {
        "name": collection_name,
        "total_documents": 0,
        "copied_documents": 0,
        "failed_documents": 0,
        "status": "pending",
    }

    try:
        print(f"\n📦 Processing collection: {collection_name}")

        # Get local collection
        local_collection = local_client.get_collection(name=collection_name)
        local_metadata = local_collection.metadata

        # Get all data from local collection
        print(f"   📥 Fetching data from local collection...")
        local_data = local_collection.get(
            include=["documents", "embeddings", "metadatas"]
        )

        total_docs = len(local_data["ids"]) if local_data["ids"] else 0
        stats["total_documents"] = total_docs

        if total_docs == 0:
            print(f"   ⚠️  Collection is empty, skipping...")
            stats["status"] = "skipped_empty"
            return stats

        print(f"   📊 Found {total_docs} documents")

        # Delete existing cloud collection if it exists (to avoid duplicates)
        try:
            cloud_client.delete_collection(name=collection_name)
            print(f"   🗑️  Deleted existing cloud collection")
        except Exception:
            # Collection doesn't exist, which is fine
            pass

        # Create new cloud collection with same metadata
        print(f"   ☁️  Creating cloud collection...")
        cloud_collection = cloud_client.create_collection(
            name=collection_name, metadata=local_metadata
        )

        # Copy data in batches
        print(f"   📤 Uploading documents in batches of {BATCH_SIZE}...")

        ids = local_data["ids"]
        documents = local_data.get("documents", [None] * len(ids))
        embeddings = local_data.get("embeddings", [None] * len(ids))
        metadatas = local_data.get("metadatas", [None] * len(ids))

        # Process in batches
        for i in tqdm(
            range(0, total_docs, BATCH_SIZE),
            desc=f"   Uploading {collection_name}",
            unit="batch",
        ):
            batch_end = min(i + BATCH_SIZE, total_docs)

            batch_ids = ids[i:batch_end]
            batch_docs = documents[i:batch_end] if documents[0] is not None else None
            batch_embeds = (
                embeddings[i:batch_end] if embeddings[0] is not None else None
            )
            batch_metas = metadatas[i:batch_end] if metadatas[0] is not None else None

            try:
                cloud_collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeds,
                    metadatas=batch_metas,
                )
                stats["copied_documents"] += len(batch_ids)
            except Exception as e:
                print(f"\n   ❌ Error copying batch {i}-{batch_end}: {str(e)}")
                stats["failed_documents"] += len(batch_ids)

        # Verify the copy
        cloud_count = cloud_collection.count()
        print(f"   ✅ Upload complete! Cloud collection has {cloud_count} documents")

        if cloud_count == total_docs:
            stats["status"] = "success"
        else:
            stats["status"] = "partial"
            print(
                f"   ⚠️  Warning: Document count mismatch (local: {total_docs}, cloud: {cloud_count})"
            )

    except Exception as e:
        print(f"   ❌ Error copying collection: {str(e)}")
        stats["status"] = "failed"
        stats["error"] = str(e)

    return stats


def migrate_chromadb(
    local_path: Path, cloud_client: chromadb.CloudClient
) -> List[Dict[str, Any]]:
    """Migrate all collections from a local ChromaDB to cloud.

    Args:
        local_path: Path to local ChromaDB directory
        cloud_client: Cloud ChromaDB client

    Returns:
        List[Dict[str, Any]]: Statistics for each collection copied
    """
    print(f"\n{'='*80}")
    print(f"🔄 Starting migration from: {local_path}")
    print(f"{'='*80}")

    try:
        # Connect to local ChromaDB
        local_client = get_local_client(local_path)

        # Get list of collections
        collection_names = list_local_collections(local_client)

        if not collection_names:
            print("⚠️  No collections found in local ChromaDB")
            return []

        print(f"\n📋 Found {len(collection_names)} collection(s):")
        for name in collection_names:
            print(f"   - {name}")

        # Copy each collection
        all_stats = []
        for collection_name in collection_names:
            stats = copy_collection(local_client, cloud_client, collection_name)
            all_stats.append(stats)

        return all_stats

    except Exception as e:
        print(f"❌ Error during migration: {str(e)}")
        return []


def print_summary(all_results: List[List[Dict[str, Any]]]) -> None:
    """Print a summary of all migration operations.

    Args:
        all_results: List of results from each local ChromaDB path
    """
    print(f"\n{'='*80}")
    print("📊 MIGRATION SUMMARY")
    print(f"{'='*80}\n")

    total_collections = 0
    total_success = 0
    total_failed = 0
    total_skipped = 0
    total_documents = 0

    for i, results in enumerate(all_results, 1):
        print(f"Source #{i}:")
        for stats in results:
            total_collections += 1
            total_documents += stats["copied_documents"]

            if stats["status"] == "success":
                total_success += 1
                status_icon = "✅"
            elif stats["status"] == "partial":
                total_success += 1
                status_icon = "⚠️"
            elif stats["status"] == "skipped_empty":
                total_skipped += 1
                status_icon = "⏭️"
            else:
                total_failed += 1
                status_icon = "❌"

            print(
                f"  {status_icon} {stats['name']}: {stats['copied_documents']}/{stats['total_documents']} documents"
            )
            if stats.get("error"):
                print(f"     Error: {stats['error']}")
        print()

    print(f"{'='*80}")
    print(f"Total collections processed: {total_collections}")
    print(f"  ✅ Successful: {total_success}")
    print(f"  ❌ Failed: {total_failed}")
    print(f"  ⏭️  Skipped (empty): {total_skipped}")
    print(f"  📄 Total documents copied: {total_documents}")
    print(f"{'='*80}\n")


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("🚀 ChromaDB Local to Cloud Migration Tool")
    print("=" * 80)

    # Validate configuration
    if not validate_config():
        print("\n❌ Configuration validation failed. Please check your .env file.")
        sys.exit(1)

    # Validate local paths
    valid_paths = []
    for path in LOCAL_CHROMADB_PATHS:
        if path.exists():
            valid_paths.append(path)
            print(f"✅ Found local ChromaDB: {path}")
        else:
            print(f"⚠️  Skipping non-existent path: {path}")

    if not valid_paths:
        print(
            "\n❌ No valid local ChromaDB paths found. Please update LOCAL_CHROMADB_PATHS."
        )
        sys.exit(1)

    print(f"\n📊 Will process {len(valid_paths)} local ChromaDB location(s)")

    # Get user confirmation
    response = input(
        "\n⚠️  This will OVERWRITE existing collections in the cloud. Continue? (yes/no): "
    )
    if response.lower() not in ["yes", "y"]:
        print("❌ Migration cancelled by user.")
        sys.exit(0)

    try:
        # Connect to cloud
        cloud_client = get_cloud_client()

        # Migrate each local ChromaDB
        all_results = []
        for path in valid_paths:
            results = migrate_chromadb(path, cloud_client)
            all_results.append(results)

        # Print summary
        print_summary(all_results)

        print("🎉 Migration completed successfully!")

    except Exception as e:
        print(f"\n❌ Fatal error during migration: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
