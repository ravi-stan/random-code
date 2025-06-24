"""
download_pdfs_from_blob.py

Read a CSV that contains a column named 'uuid'.  
For every UUID:
1. Build the blob path  document-<uuid>/document_api.blob
2. Connect to Azure Blob Storage.
3. Check the blob's Content-Type (or magic header) to confirm it is a PDF.
4. If it is a PDF, download it to <target_dir>/<uuid>.pdf.
"""

from pathlib import Path
import os
import mimetypes
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError

# ──────────────────────────────────────────────────────────────────────────────
# Configuration – edit these to match your environment
CSV_PATH        = "uuids.csv"              # CSV file with a 'uuid' column
CONTAINER_NAME  = "my-container"           # Blob container
TARGET_DIR      = Path("downloaded_pdfs")  # Local directory to save PDFs
# Either use an account URL + credential OR a connection string
AZURE_CONN_STR  = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# ──────────────────────────────────────────────────────────────────────────────
def is_pdf_content_type(content_type: str | None) -> bool:
    """Return True if Content-Type looks like PDF."""
    if not content_type:
        return False
    return content_type.lower() in {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",  # sometimes used with unknown types
    }

def header_looks_like_pdf(stream) -> bool:
    """
    Download the first 5 bytes and verify it starts with '%PDF-'.
    `stream` is an azure.storage.blob.BlobClient or DownloadStream.
    """
    head = stream.read(5)
    stream.seek(0)  # reset so we can read again later
    return head == b"%PDF-"

def download_pdfs(
    csv_path: str | Path,
    container_name: str,
    target_dir: Path,
    conn_str: str,
) -> None:
    df = pd.read_csv(csv_path, dtype={"uuid": str})
    target_dir.mkdir(parents=True, exist_ok=True)

    service = BlobServiceClient.from_connection_string(conn_str)
    container = service.get_container_client(container_name)

    for uuid in df["uuid"].dropna().unique():
        blob_path = f"document-{uuid}/document_api.blob"
        blob = container.get_blob_client(blob_path)

        try:
            props = blob.get_blob_properties()
        except ResourceNotFoundError:
            print(f"Blob not found: {blob_path}")
            continue
        except HttpResponseError as e:
            print(f"Error fetching properties for {blob_path}: {e}")
            continue

        # First check Content-Type header (cheap)
        if not is_pdf_content_type(props.content_settings.content_type):
            print(f"Skipped (Content-Type not PDF): {blob_path}")
            continue

        # Optional: double-check magic header (costs one small read)
        stream = blob.download_blob(offset=0, length=5)
        if not header_looks_like_pdf(stream):
            print(f"Skipped (header not PDF): {blob_path}")
            continue

        # Now download the whole blob
        download_stream = blob.download_blob()
        dest_file = target_dir / f"{uuid}.pdf"
        with open(dest_file, "wb") as fh:
            download_stream.readinto(fh)
        print(f"Saved {dest_file}")

if __name__ == "__main__":
    download_pdfs(
        csv_path=CSV_PATH,
        container_name=CONTAINER_NAME,
        target_dir=TARGET_DIR,
        conn_str=AZURE_CONN_STR,
    )

