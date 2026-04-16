
"""
Remove all collections in a Qdrant instance.

Environment variables supported:
  QDRANT_URL        - e.g., http://localhost:6333 or https://YOUR-CLOUD-ENDPOINT
  QDRANT_API_KEY    - required for Qdrant Cloud / secured instances
  QDRANT_TIMEOUT    - optional, in seconds (default: 60)
  DRY_RUN           - set to "1" to only list collections without deletion
"""

import os
from qdrant_client import QdrantClient

def main():
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    timeout = float(os.getenv("QDRANT_TIMEOUT", "60"))
    dry_run = os.getenv("DRY_RUN", "0") == "1"

    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=False,  # set True if you prefer gRPC and the endpoint supports it
    )

    # List collections
    resp = client.get_collections()
    collections = [c.name for c in resp.collections]

    if not collections:
        print("No collections found.")
        return

    print(f"Found {len(collections)} collections:")
    for name in collections:
        print(f"  - {name}")

    if dry_run:
        print("\nDRY_RUN is enabled. No deletions performed.")
        return

    print("\nDeleting collections...")
    failures = []

    for name in collections:
        try:
            # delete_collection returns immediately; if the collection doesn't exist,
            # the client may raise an exception (depending on version).
            client.delete_collection(collection_name=name)
            print(f"✓ Deleted: {name}")
        except Exception as e:
            # Catch-all to keep the batch running; log failures for review.
            print(f"✗ Failed: {name} — {e}")
            failures.append((name, str(e)))

    if failures:
        print("\nSome deletions failed:")
        for name, err in failures:
            print(f"  {name}: {err}")
        raise SystemExit(1)
    else:
        print("\nAll collections deleted successfully.")

if __name__ == "__main__":
    main()