from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContainerClient


# --------------------------------------------------------------------------- #
# Configuration constants – you can also inject these via env vars or DI
# --------------------------------------------------------------------------- #
ACCOUNT_URL = "https://<your‑account>.blob.core.windows.net"      # no trailing slash
CONTAINER   = "<container-name>"                                  # e.g. "raw-documents"
# A SAS token, an account key, or DefaultAzureCredential() object all work:
CREDENTIAL  = "<sas|key|DefaultAzureCredential()>"


# --------------------------------------------------------------------------- #
# Main helper
# --------------------------------------------------------------------------- #
TSV_PATTERN = re.compile(
    r"^document-(?P<doc>[^/]+)/pipeline-(?P<pipe>[^/]+)/(?P=pipe)\.document_reassembly\.tsv$"
)

def download_reassembly_tsv(
    document_uuid: str,
    *,
    dest: os.PathLike | str | None = None,
    account_url: str = ACCOUNT_URL,
    credential=CREDENTIAL,
    container: str = CONTAINER,
) -> Path | bytes:
    """
    Download the *sole* document‑reassembly TSV belonging to ``document_uuid``.

    Parameters
    ----------
    document_uuid : str
        The document’s public UUID (without braces).
    dest : os.PathLike | str | None, optional
        If supplied, the file is written here and the path is returned.
        If *None*, the blob is read into memory and the raw bytes are returned.
    account_url, credential, container : str
        Override the storage location/credentials if desired.

    Raises
    ------
    FileNotFoundError
        If no matching TSV exists.
    RuntimeError
        If more than one match is found (shouldn’t occur in a well‑formed bucket).

    Examples
    --------
    >>> path = download_reassembly_tsv("0d3b5c3e‑55c2‑4d96‑8e7d‑7da9ba9a6b13",
    ...                                dest="out/doc.tsv",
    ...                                credential=my_sas)
    >>> print(f"Saved to {path}")
    """
    # 1) Connect
    blob_service = BlobServiceClient(account_url=account_url, credential=credential)
    container_client: ContainerClient = blob_service.get_container_client(container)

    # 2) Narrow the search with a prefix (saves API calls & bandwidth)
    prefix = f"document-{document_uuid}/"
    candidates = [
        b.name
        for b in container_client.walk_blobs(name_starts_with=prefix)
        if TSV_PATTERN.fullmatch(b.name)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No *.document_reassembly.tsv found for UUID {document_uuid!r}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Expected exactly one TSV for UUID {document_uuid!r}, found {len(candidates)}: {candidates}"
        )

    blob_name = candidates[0]
    blob_client = container_client.get_blob_client(blob_name)

    # 3) Download
    stream = blob_client.download_blob(max_concurrency=4)
    data = stream.readall()

    # 4) Persist or return
    if dest is None:
        return data

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)
    return dest_path

