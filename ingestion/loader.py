
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(folder_path):
    folder = Path(folder_path).resolve()

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    all_documents = []

    for pdf_path in pdf_files:
        documents = PyPDFLoader(str(pdf_path)).load()

        for page in documents:
            page.metadata["source"] = pdf_path.name
            page.metadata["document_name"] = pdf_path.name
            page.metadata["page_number"] = (
                int(page.metadata.get("page", 0)) + 1
            )

        all_documents.extend(documents)

        print(f"Loaded {len(documents)} pages from {pdf_path.name}")

    print(f"\nTotal PDFs loaded: {len(pdf_files)}")
    print(f"Total pages loaded: {len(all_documents)}")

    return all_documents
