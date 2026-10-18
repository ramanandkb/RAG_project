from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]

    print(f"Created {len(chunks)} chunks")
    return chunks