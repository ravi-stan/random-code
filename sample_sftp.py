"""
Fast loader: read <uuid> list from a CSV, build blob paths, then download the
corresponding blobs in parallel **into memory** as a list of dictionaries:
    [{"<path>": <bytes>}, …]

Requirements
------------
pip install "azure-storage-blob[aio]>=12.19" pandas  # or replace pandas with csv

Environment
-----------
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=...;AccountKey=..."
"""

from __future__ import annotations
import asyncio
import os
from typing import Callable, List, Dict

import pandas as pd
from azure.storage.blob.aio import BlobServiceClient

# ---------- 1. Helpers ------------------------------------------------------- #
def read_uuid_column(csv_path: str, column: str = "uuid") -> list[str]:
    """Return the list of UUID strings from `column` in `csv_path`."""
    return pd.read_csv(csv_path, usecols=[column], dtype=str)[column].tolist()

def make_path(template: Callable[[str], str], uuid_: str) -> str:
    """Apply user-supplied template to one uuid to build the blob path."""
    return template(uuid_)

async def _fetch_one(
    svc: BlobServiceClient,
    container: str,
    blob_path: str,
    sem: asyncio.Semaphore,
) -> dict[str, bytes]:
    """
    Download a single blob as bytes.  Returns {blob_path: data}.
    A semaphore limits overall concurrency.
    """
    async with sem:                               # bound concurrency
        blob_client = svc.get_blob_client(container, blob_path)
        stream = await blob_client.download_blob()
        data: bytes = await stream.readall()
    return {blob_path: data}

# ---------- 2. Orchestrator -------------------------------------------------- #
async def load_blobs_from_csv(
    csv_path: str,
    container: str,
    path_template: Callable[[str], str] | str,
    *,
    concurrency: int = 64,
) -> List[Dict[str, bytes]]:
    """
    Read UUIDs → build paths → download each blob concurrently.
    Returns a **list of dictionaries** so the caller preserves ordering if desired.
    """
    # resolve template to a callable once
    if isinstance(path_template, str):
        tmpl: str = path_template
        path_template = lambda u, t=tmpl: t.format(uuid=u)          # noqa: E731

    uuids = read_uuid_column(csv_path)
    blob_paths = [make_path(path_template, u) for u in uuids]

    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    svc = BlobServiceClient.from_connection_string(conn_str)

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_fetch_one(svc, container, p, sem))
        for p in blob_paths
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    await svc.close()
    return results

# ---------- 3. Convenient sync wrapper -------------------------------------- #
def load_blobs(
    csv_path: str,
    container: str,
    path_template: Callable[[str], str] | str,
    *,
    concurrency: int = 64,
) -> List[Dict[str, bytes]]:
    """Synchronous façade so callers needn’t manage an event-loop."""
    return asyncio.run(
        load_blobs_from_csv(csv_path, container, path_template, concurrency=concurrency)
    )

# ---------- 4. Example usage ------------------------------------------------- #
if __name__ == "__main__":
    # Example blob path template:
    #   f"raw/{uuid[:2]}/{uuid}.json"   ← stash by first two chars for scale
    blobs = load_blobs(
        csv_path="uuids.csv",
        container="my-container",
        path_template=lambda u: f"raw/{u[:2]}/{u}.json",
        concurrency=128,                # tune to your network + cores
    )
    # blobs is now a list like: [{"raw/ab/ab…json": b"..."} , …]

