"""Test ChromaDB Cloud/Local Switching

This script tests the chromadb_client_factory functionality to ensure
it correctly switches between cloud and local ChromaDB based on environment configuration.

Usage:
    # Test with cloud (set CHROMA_USE_CLOUD=true in .env)
    python test_chromadb_switching.py

    # Test with local (set CHROMA_USE_CLOUD=false in .env)
    python test_chromadb_switching.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
try:
    BASE_DIR = Path(__file__).resolve().parents[0]
except NameError:
    BASE_DIR = Path(os.getcwd())

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from metadata.chromadb_client_factory import (
    should_use_cloud,
    get_chromadb_client,
    get_chromadb_collection,
)


def test_configuration():
    """Test the environment configuration."""
    print("=" * 80)
    print("🧪 ChromaDB Cloud/Local Switching Test")
    print("=" * 80)

    # Check configuration
    use_cloud = should_use_cloud()
    chroma_use_cloud = os.getenv("CHROMA_USE_CLOUD", "false")

    print(f"\n📋 Configuration:")
    print(f"   CHROMA_USE_CLOUD: {chroma_use_cloud}")
    print(f"   Resolved to: {'Cloud' if use_cloud else 'Local'}")

    if use_cloud:
        api_key = os.getenv("CHROMA_API_KEY", "").strip("',\"")
        tenant = os.getenv("CHROMA_API_TENANT", "").strip("',\"")
        database = os.getenv("CHROMA_API_DATABASE", "").strip("',\"")

        print(f"\n🌐 Cloud Configuration:")
        print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
        print(f"   Tenant: {tenant if tenant else '❌ Missing'}")
        print(f"   Database: {database if database else '❌ Missing'}")
    else:
        print(f"\n📂 Local Configuration:")
        print(f"   Will use local PersistentClient")

    return use_cloud


def test_client_creation():
    """Test client creation."""
    print(f"\n{'=' * 80}")
    print("🔧 Testing Client Creation")
    print("=" * 80)

    use_cloud = should_use_cloud()

    try:
        if use_cloud:
            # Test cloud client
            print("\n🌐 Testing Cloud Client...")
            client = get_chromadb_client(
                local_path=None,  # Not needed for cloud
                collection_name="test_collection",
            )
            print(f"✅ Cloud client created successfully!")
            print(f"   Client type: {type(client).__name__}")

        else:
            # Test local client
            print("\n📂 Testing Local Client...")
            local_path = BASE_DIR / "metadata" / "czsu_chromadb"

            if not local_path.exists():
                print(f"⚠️  Local ChromaDB path does not exist: {local_path}")
                print(
                    f"   This is expected if you haven't created local collections yet."
                )
                return False

            client = get_chromadb_client(
                local_path=local_path, collection_name="czsu_selections_chromadb"
            )
            print(f"✅ Local client created successfully!")
            print(f"   Client type: {type(client).__name__}")
            print(f"   Path: {local_path}")

        return True

    except Exception as exc:
        print(f"❌ Error creating client: {str(exc)}")
        import traceback

        traceback.print_exc()
        return False


def test_collection_access():
    """Test collection access."""
    print(f"\n{'=' * 80}")
    print("📦 Testing Collection Access")
    print("=" * 80)

    use_cloud = should_use_cloud()

    try:
        if use_cloud:
            # Test cloud collection
            print("\n🌐 Testing Cloud Collection Access...")
            print("   Attempting to list collections...")

            client = get_chromadb_client(local_path=None)
            collections = client.list_collections()

            print(f"✅ Found {len(collections)} collection(s) in cloud:")
            for col in collections:
                print(f"   - {col.name}")

            if collections:
                # Try to access first collection
                col_name = collections[0].name
                print(f"\n   Accessing collection: {col_name}")
                collection = client.get_collection(name=col_name)
                count = collection.count()
                print(f"   ✅ Collection has {count} documents")

        else:
            # Test local collection
            print("\n📂 Testing Local Collection Access...")
            local_path = BASE_DIR / "metadata" / "czsu_chromadb"
            collection_name = "czsu_selections_chromadb"

            if not local_path.exists():
                print(f"⚠️  Local ChromaDB path does not exist: {local_path}")
                return False

            print(f"   Accessing collection: {collection_name}")
            collection = get_chromadb_collection(
                collection_name=collection_name, local_path=local_path
            )

            count = collection.count()
            print(f"✅ Collection accessed successfully!")
            print(f"   Collection has {count} documents")

        return True

    except Exception as exc:
        print(f"❌ Error accessing collection: {str(exc)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")

    # Test 1: Configuration
    use_cloud = test_configuration()

    # Test 2: Client Creation
    client_ok = test_client_creation()

    # Test 3: Collection Access
    if client_ok:
        collection_ok = test_collection_access()
    else:
        collection_ok = False

    # Summary
    print(f"\n{'=' * 80}")
    print("📊 Test Summary")
    print("=" * 80)
    print(f"   Mode: {'🌐 Cloud' if use_cloud else '📂 Local'}")
    print(f"   Client Creation: {'✅ Success' if client_ok else '❌ Failed'}")
    print(f"   Collection Access: {'✅ Success' if collection_ok else '❌ Failed'}")

    if client_ok and collection_ok:
        print(
            f"\n🎉 All tests passed! ChromaDB is working correctly in {'cloud' if use_cloud else 'local'} mode."
        )
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")

    print("=" * 80)


if __name__ == "__main__":
    main()
