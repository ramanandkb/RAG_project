
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from app.config import PINECONE_INDEX_NAME
from app.embeddings import get_embeddings
from app.llm import get_llm

load_dotenv()

NAMESPACE = os.getenv("PINECONE_NAMESPACE", "worldbank-docs")

SYSTEM_PROMPT = """
You are a World Bank Research Assistant.
Answer using only the provided context.
Do not invent facts or use outside knowledge.
If the answer is unavailable, say:
"I cannot find the answer in the provided knowledge base."
Cite source filenames and page numbers when available.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])


@st.cache_resource
def initialize_rag():
    embeddings = get_embeddings()
    llm = get_llm()

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE
    )
    return vector_store, llm


st.set_page_config(
    page_title="World Bank Research Assistant",
    page_icon="📚"
)

st.title("📚 World Bank Research Assistant")
st.caption("Ask questions about your uploaded World Bank reports.")

try:
    vector_store, llm = initialize_rag()
except Exception:
    st.error(
        "Could not initialize the RAG system. "
        "Check your app secrets and deployment logs."
    )
    st.stop()

question = st.chat_input("Ask a question about the reports...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching reports and generating an answer..."):
            try:
                docs = vector_store.similarity_search(
                    query=question,
                    k=4
                )

                if not docs:
                    st.write(
                        "I cannot find the answer in the "
                        "provided knowledge base."
                    )
                else:
                    context_parts = []
                    sources = []

                    for doc in docs:
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

                        item = {"source": source, "page": page}
                        if item not in sources:
                            sources.append(item)

                    messages = prompt.format_messages(
                        question=question,
                        context="\n\n".join(context_parts)
                    )

                    response = llm.invoke(messages)
                    st.markdown(response.content)

                    with st.expander("Sources"):
                        for source in sources:
                            st.write(
                                f"- {source['source']} "
                                f"(page {source['page']})"
                            )

            except Exception:
                st.error(
                    "Something went wrong while processing "
                    "your question. Check the app logs."
                )
