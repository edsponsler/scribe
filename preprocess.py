import os
import json
import tempfile
import hashlib
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.cloud import storage, exceptions
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv

"""
This script is a data processing pipeline for preparing a corpus of documents
for a Retrieval-Augmented Generation (RAG) system. It performs the following
steps:

1.  **Ingestion**: Reads raw documents (e.g., .pdf, .txt) from a specified
    Google Cloud Storage (GCS) bucket.

2.  **Processing**: For each document, it:
    a.  Downloads the file to a secure, temporary location.
    b.  Uses LangChain's document loaders to extract text content based on the
        file type.
    c.  Applies a document-aware chunking strategy using a
        RecursiveCharacterTextSplitter. A special, fine-grained chunking is
        applied to foundational texts, while a standard size is used for others.

3.  **Structuring**: Each text chunk is formatted into a JSON object containing
    the `content` and `source` metadata (the original filename). This metadata
    is crucial for accurate citations in the final application.

4.  **Output**: All processed chunks are aggregated into a single JSONL
    (JSON Lines) file named `processed_corpus.jsonl`.

5.  **Idempotent Upload**: The script calculates an MD5 hash of the newly
    generated corpus and compares it to the hash of the existing file in the
    processed GCS bucket. The upload is skipped if the content has not changed,
    preventing unnecessary file operations and re-indexing by downstream
    services like Vertex AI Search.
"""

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
RAW_DATA_BUCKET = os.getenv("RAW_DATA_BUCKET")
PROCESSED_DATA_BUCKET = os.getenv("PROCESSED_DATA_BUCKET")
# ---------------------

def process_document(blob, bucket_name):
    """
    Downloads a file from GCS, determines its type, and uses the appropriate
    document loader and chunking strategy.
    """
    # Use a context manager for robust temporary file handling
    _, file_extension = os.path.splitext(blob.name)
    with tempfile.NamedTemporaryFile(suffix=file_extension) as temp_file:
        file_path = temp_file.name
        blob.download_to_filename(file_path)

        print(f"Processing document: {blob.name}")

        # Document-aware loading based on file extension
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() == ".txt":
            loader = TextLoader(file_path)
        else:
            print(f"Skipping unsupported file type: {blob.name}")
            return []

        documents = loader.load()

        # Document-aware chunking strategy as per the ASRR Plan [cite: 58]
        if "emergent_intelligence_framework" in blob.name:
            # Special, fine-grained chunking for the foundational text [cite: 65]
            print("-> Applying special chunking for foundational text.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        else:
            # Standard chunking for scientific texts and technical docs [cite: 59, 62]
            print("-> Applying standard chunking.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

        chunks = text_splitter.split_documents(documents)

        return chunks

def main():
    """
    Main function to orchestrate the preprocessing pipeline.
    """
    # Validate that environment variables are set
    if not all([GCP_PROJECT_ID, RAW_DATA_BUCKET, PROCESSED_DATA_BUCKET]):
        print("Error: Missing required environment variables.")
        print("Please create a .env file from the .env-example template")
        print("and fill in your GCP_PROJECT_ID, RAW_DATA_BUCKET, and PROCESSED_DATA_BUCKET.")
        return

    storage_client = storage.Client(project=GCP_PROJECT_ID)
    raw_bucket = storage_client.bucket(RAW_DATA_BUCKET)
    
    all_processed_chunks = []

    # Iterate through all files in the raw data bucket
    for blob in raw_bucket.list_blobs():
        # Get the text chunks from the document
        chunks = process_document(blob, RAW_DATA_BUCKET)

        # For each chunk, create a JSON object with content and metadata
        for chunk in chunks:
            # This step is critical for accurate citations later 
            chunk_json = {
                "content": chunk.page_content,
                "source": blob.name  # Metadata extraction [cite: 68]
            }
            all_processed_chunks.append(json.dumps(chunk_json))

    if not all_processed_chunks:
        print("No documents were processed. Exiting.")
        return

    # Combine all JSON strings into a single JSONL file content
    jsonl_content = "\n".join(all_processed_chunks)
    
    # --- UPLOAD LOGIC ---
    # Only upload if the content has actually changed to avoid unnecessary re-indexing.
    processed_bucket = storage_client.bucket(PROCESSED_DATA_BUCKET)
    output_blob_name = "processed_corpus.jsonl"
    output_blob = processed_bucket.blob(output_blob_name)

    # Calculate MD5 hash of the new content to check for changes.
    new_content_bytes = jsonl_content.encode('utf-8')
    new_hash = base64.b64encode(hashlib.md5(new_content_bytes).digest()).decode('utf-8')

    try:
        # Get blob metadata without downloading the whole file.
        output_blob.reload()
        if output_blob.md5_hash == new_hash:
            print("\n✅ No changes detected in corpus. Upload skipped.")
            return
    except exceptions.NotFound:
        # The blob doesn't exist, so we must upload.
        print("\nCorpus file not found. Creating new one...")
        pass

    print(f"\nUploading {len(all_processed_chunks)} chunks to gs://{PROCESSED_DATA_BUCKET}/{output_blob_name}...")
    output_blob.upload_from_string(new_content_bytes, content_type="application/jsonl")
    print("✅ Preprocessing and upload complete.")


if __name__ == "__main__":
    main()