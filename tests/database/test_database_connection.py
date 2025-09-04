import os
import sys
import asyncio
import time
from dotenv import load_dotenv

# Windows event loop fix for PostgreSQL compatibility
if sys.platform == "win32":
    print(
        "[POSTGRES-STARTUP] Windows detected - setting SelectorEventLoop for PostgreSQL compatibility..."
    )
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[POSTGRES-STARTUP] Event loop policy set successfully")

# Load environment variables
load_dotenv()


def test_sync_connection():
    """Test synchronous database connection using psycopg"""
    try:
        import psycopg

        # Get connection parameters from environment
        password = os.getenv("password")
        user = os.getenv("user")
        host = os.getenv("host")
        port = os.getenv("port", 6543)
        dbname = os.getenv("dbname")

        # Validate required parameters
        if not all([password, user, host, dbname]):
            print("❌ Missing required connection parameters")
            print(f"User: {user}")
            print(f"Host: {host}")
            print(f"Port: {port}")
            print(f"Database: {dbname}")
            print(f"Password: {'***' if password else 'None'}")
            return False

        # Build connection string
        connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        )

        print("🧪 Testing synchronous database connection...")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Database: {dbname}")
        print(f"User: {user}")

        start_time = time.time()

        # Test connection
        with psycopg.connect(connection_string) as conn:
            with conn.cursor() as cur:
                # Test basic query
                cur.execute(
                    "SELECT version(), current_database(), current_user, now();"
                )
                result = cur.fetchone()

                end_time = time.time()

                print("✅ Synchronous connection successful!")
                print(f"📊 Connection time: {end_time - start_time:.2f} seconds")
                print(f"🗄️  PostgreSQL Version: {result[0].split(',')[0]}")
                print(f"📁 Database: {result[1]}")
                print(f"👤 User: {result[2]}")
                print(f"🕐 Server Time: {result[3]}")

                # Test table query (optional)
                try:
                    cur.execute(
                        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
                    )
                    table_count = cur.fetchone()[0]
                    print(f"📋 Public tables count: {table_count}")
                except Exception as e:
                    print(f"⚠️  Could not count tables: {e}")

                return True

    except ImportError:
        print("❌ psycopg library not found. Install with: pip install psycopg[binary]")
        return False
    except Exception as e:
        print(f"❌ Synchronous connection failed: {e}")
        return False


async def test_async_connection():
    """Test asynchronous database connection using asyncpg"""
    try:
        import asyncpg

        # Get connection parameters from environment
        password = os.getenv("password")
        user = os.getenv("user")
        host = os.getenv("host")
        port = int(os.getenv("port", 6543))
        dbname = os.getenv("dbname")

        # Validate required parameters
        if not all([password, user, host, dbname]):
            print("❌ Missing required connection parameters for async test")
            return False

        print("\n🧪 Testing asynchronous database connection...")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Database: {dbname}")
        print(f"User: {user}")

        start_time = time.time()

        # Test connection
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=dbname,
            host=host,
            port=port,
            ssl="require",
        )

        try:
            # Test basic query
            result = await conn.fetchrow(
                "SELECT version(), current_database(), current_user, now();"
            )

            end_time = time.time()

            print("✅ Asynchronous connection successful!")
            print(f"📊 Connection time: {end_time - start_time:.2f} seconds")
            print(f"🗄️  PostgreSQL Version: {result['version'].split(',')[0]}")
            print(f"📁 Database: {result['current_database']}")
            print(f"👤 User: {result['current_user']}")
            print(f"🕐 Server Time: {result['now']}")

            # Test table query (optional)
            try:
                table_count = await conn.fetchval(
                    "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
                )
                print(f"📋 Public tables count: {table_count}")
            except Exception as e:
                print(f"⚠️  Could not count tables: {e}")

            return True

        finally:
            await conn.close()

    except ImportError:
        print("❌ asyncpg library not found. Install with: pip install asyncpg")
        return False
    except Exception as e:
        print(f"❌ Asynchronous connection failed: {e}")
        return False


async def test_psycopg_async():
    """Test asynchronous database connection using psycopg async"""
    try:
        import psycopg

        # Get connection parameters from environment
        password = os.getenv("password")
        user = os.getenv("user")
        host = os.getenv("host")
        port = os.getenv("port", 6543)
        dbname = os.getenv("dbname")

        # Validate required parameters
        if not all([password, user, host, dbname]):
            print("❌ Missing required connection parameters for psycopg async test")
            return False

        # Build connection string
        connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        )

        print("\n🧪 Testing psycopg asynchronous database connection...")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Database: {dbname}")
        print(f"User: {user}")
        print("🔄 Attempting psycopg async connection...")

        start_time = time.time()

        # Test connection with timeout
        try:
            connection_task = psycopg.AsyncConnection.connect(connection_string)
            conn = await asyncio.wait_for(connection_task, timeout=10.0)
            print("✅ Psycopg async connection established!")
        except asyncio.TimeoutError:
            print("❌ Psycopg async connection timed out after 10 seconds")
            return False

        async with conn:
            async with conn.cursor() as cur:
                print("🔄 Executing psycopg async query...")
                # Test basic query with timeout
                query_task = cur.execute(
                    "SELECT version(), current_database(), current_user, now();"
                )
                await asyncio.wait_for(query_task, timeout=5.0)
                result = await cur.fetchone()

                end_time = time.time()

                print("✅ Psycopg asynchronous connection successful!")
                print(f"📊 Connection time: {end_time - start_time:.2f} seconds")
                print(f"🗄️  PostgreSQL Version: {result[0].split(',')[0]}")
                print(f"📁 Database: {result[1]}")
                print(f"👤 User: {result[2]}")
                print(f"🕐 Server Time: {result[3]}")

                # Test table query (optional)
                try:
                    await asyncio.wait_for(
                        cur.execute(
                            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
                        ),
                        timeout=5.0,
                    )
                    table_count = await cur.fetchone()
                    print(f"📋 Public tables count: {table_count[0]}")
                except asyncio.TimeoutError:
                    print("⚠️  Table count query timed out")
                except Exception as e:
                    print(f"⚠️  Could not count tables: {e}")

                return True

    except asyncio.TimeoutError:
        print("❌ Psycopg async operation timed out")
        return False
    except Exception as e:
        print(f"❌ Psycopg asynchronous connection failed: {e}")
        return False


def test_connection_pool():
    """Test connection pooling"""
    try:
        import psycopg_pool

        # Get connection parameters from environment
        password = os.getenv("password")
        user = os.getenv("user")
        host = os.getenv("host")
        port = os.getenv("port", 6543)
        dbname = os.getenv("dbname")

        # Validate required parameters
        if not all([password, user, host, dbname]):
            print("❌ Missing required connection parameters for pool test")
            return False

        # Build connection string
        connection_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
        )

        print("\n🏊 Testing connection pool...")
        print("🔄 Creating connection pool...")

        # Create connection pool with timeout
        try:
            pool = psycopg_pool.ConnectionPool(
                connection_string,
                min_size=1,
                max_size=3,
                open=True,
                timeout=10.0,  # 10 second timeout for pool operations
            )
            print("✅ Connection pool created successfully!")
        except Exception as e:
            print(f"❌ Failed to create connection pool: {e}")
            return False

        try:
            start_time = time.time()
            print("🔄 Getting connection from pool...")

            # Test connection from pool with timeout
            with pool.connection(timeout=5.0) as conn:
                print("✅ Got connection from pool!")
                with conn.cursor() as cur:
                    print("🔄 Executing pool query...")
                    cur.execute(
                        "SELECT 'Pool connection successful!' as message, now();"
                    )
                    result = cur.fetchone()

                    end_time = time.time()

                    print("✅ Connection pool test successful!")
                    print(
                        f"📊 Pool connection time: {end_time - start_time:.2f} seconds"
                    )
                    print(f"💬 Message: {result[0]}")
                    print(f"🕐 Server Time: {result[1]}")
                    print(f"🏊 Pool stats - Size: {pool.get_stats()}")

                    return True

        except Exception as e:
            print(f"❌ Pool connection test failed: {e}")
            return False
        finally:
            print("🔄 Closing connection pool...")
            pool.close()
            print("✅ Connection pool closed!")

    except ImportError:
        print(
            "❌ psycopg_pool library not found. Install with: pip install psycopg-pool"
        )
        return False
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False


async def main():
    """Main function to run all tests"""
    print("🚀 Starting database connection tests...")
    print("=" * 50)

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found in current directory")
        print("Make sure you're running this script from the project root")
        return

    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🔧 .env file found: {os.path.exists('.env')}")

    # Test synchronous connection
    print("\n🔄 Starting synchronous connection test...")
    sync_success = test_sync_connection()

    # Test asynchronous connections
    if sync_success:
        print("\n🔄 Starting asyncpg connection test...")
        try:
            await asyncio.wait_for(test_async_connection(), timeout=30.0)
        except asyncio.TimeoutError:
            print("❌ Asyncpg test timed out after 30 seconds")

        print("\n🔄 Starting psycopg async connection test...")
        try:
            await asyncio.wait_for(test_psycopg_async(), timeout=30.0)
        except asyncio.TimeoutError:
            print("❌ Psycopg async test timed out after 30 seconds")

        print("\n🔄 Starting connection pool test...")
        try:
            # Run pool test in executor since it's not async
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, test_connection_pool), timeout=30.0
            )
        except asyncio.TimeoutError:
            print("❌ Connection pool test timed out after 30 seconds")
    else:
        print("⚠️  Skipping async tests due to sync connection failure")

    print("\n" + "=" * 50)
    print("🏁 Database connection tests completed!")


if __name__ == "__main__":
    # Run the tests
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
