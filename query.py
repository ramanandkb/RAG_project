from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from app.config import PINECONE_INDEX_NAME
from app.embeddings import get_embeddings
from app.llm import get_llm

import os

load_dotenv()

NAMESPACE = os.getenv("PINECONE_NAMESPACE", "worldbank-docs")

SYSTEM_PROMPT = """
You are a World Bank Research Assistant.

Answer the user's question using ONLY the provided context.

Instructions:
1. Give a direct, natural-language answer.
2. Never display raw document chunks or similarity scores.
3. Do not invent facts or use outside knowledge.
4. If the context does not contain the answer, say:
   "I cannot find the answer in the provided knowledge base."
5. Mention the source filename and page number when available.
6. If the context is insufficient to answer confidently,
   clearly explain the limitation instead of guessing.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


def main():
    # Initialize embedding model and LLM
    embeddings = get_embeddings()
    llm = get_llm()

    # Connect to the same Pinecone index and namespace used during ingestion
    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE
    )
    while True:
        question = input("Ask your question \nYou:").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting the application.")
            break

        if not question:
            print("Please enter a question.")
            return

        # Retrieve relevant document chunks
        results = vector_store.similarity_search(
            query=question,
            k=4
        )

        if not results:
            print("I cannot find the answer in the provided knowledge base.")
            return

        # Build context for the LLM
        context_parts = []

        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get(
                "page_number",
                doc.metadata.get("page", "Unknown")
            )

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # Generate a natural-language answer
        messages = prompt.format_messages(
            question=question,
            context=context
        )

        response = llm.invoke(messages)

        print("\n========== ANSWER ==========\n")
        print(response.content)
        print("\n============================")


if __name__ == "__main__":
    main()