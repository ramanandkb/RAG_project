from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.vectorstore import get_vector_store
from app.llm import get_llm
from app.prompts import get_prompt
from app.config import TOP_K


def create_rag_chain():

    # Vector database
    vector_store = get_vector_store()

    # Retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    # LLM
    llm = get_llm()

    # Prompt
    prompt = get_prompt()

    # Combine retrieved documents + LLM
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Complete RAG pipeline
    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    return rag_chain