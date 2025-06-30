import sys
import os
import asyncio
from chromadb import PersistentClient

# Fix path so we can import app.*
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.vector_service import index_examples, COLLECTION, CHROMA_PATH

async def main():
    # Prepare test input
    test_data = [
        {"id": "test-001", "text": "This is a test invoice example."}
    ]

    # Run indexer
    await index_examples(test_data)

    # Check if it got stored
    client = PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)

    print("\n✅ Chroma check:")
    print(f"Count: {col.count()}")
    print(f"Peek: {col.peek()}")

    # Cleanup test entry
    col.delete(ids=["test-001"])
    print("🧹 Cleanup done.")

if __name__ == "__main__":
    asyncio.run(main())
