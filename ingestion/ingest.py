from pathlib import Path

from ingestion.loader import load_pdf
from ingestion.chunker import split_documents
from app.embeddings import get_embeddings
from app.config import PINECONE_INDEX_NAME

from langchain_pinecone import PineconeVectorStore


DOCUMENTS_DIR = Path("documents")


def add_metadata(chunks, file_path):

    file_name = Path(file_path).name

    for index, chunk in enumerate(chunks):

        chunk.metadata["document_name"] = file_name
        chunk.metadata["chunk_id"] = index

    return chunks


def ingest_pdf(file_path):

    print("\n==============================")
    print(f"Processing: {file_path}")
    print("==============================")

    # Load
    documents = load_pdf(file_path)

    # Chunk
    chunks = split_documents(documents)

    # Metadata
    chunks = add_metadata(
        chunks,
        file_path
    )

    # Embeddings
    embeddings = get_embeddings()

    # Pinecone
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Store
    vector_store.add_documents(chunks)

    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("Ingestion completed!")


def main():

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        ingest_pdf(pdf_file)


if __name__ == "__main__":
    main()