
from pathlib import Path
import hashlib
import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from app.config import PINECONE_INDEX_NAME
from app.embeddings import get_embeddings
from ingestion.loader import load_pdfs
from ingestion.chunker import split_documents


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT_DIR / "documents"

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

# Use the default namespace if none is configured
NAMESPACE = os.getenv("PINECONE_NAMESPACE") or None


def generate_chunk_id(chunk):
    """Generate a stable ID for each document chunk."""
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page_number", 1)
    start = chunk.metadata.get("start_index", 0)

    unique_text = (
        f"{source}|{page}|{start}|{chunk.page_content}"
    )

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


def main():
    print("Starting World Bank RAG ingestion...")

    # 1. Load all PDF documents
    documents = load_pdfs(DOCUMENTS_DIR)

    # 2. Split documents into chunks
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"\nTotal pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    # 3. Generate stable IDs and preserve metadata
    ids = []

    for chunk in chunks:
        chunk.metadata["chunk_id"] = generate_chunk_id(chunk)
        ids.append(chunk.metadata["chunk_id"])

    # 4. Initialize the embedding model
    print("\nLoading embedding model...")
    embeddings = get_embeddings()

    # 5. Connect to the existing Pinecone index
    print(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    # 6. Upload document chunks to Pinecone
    print("\nUploading chunks to Pinecone...")

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch_documents = chunks[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        uploaded = min(start + len(batch_documents), len(chunks))
        print(f"Uploaded {uploaded}/{len(chunks)} chunks")

    print("\n========== INGESTION COMPLETED ==========")
    print(f"PDF pages processed: {len(documents)}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Pinecone index: {PINECONE_INDEX_NAME}")
    print(f"Namespace: {NAMESPACE or '__default__'}")


if __name__ == "__main__":
    main()
