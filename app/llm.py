from langchain_groq import ChatGroq

from app.config import LLM_MODEL


def get_llm():

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.2
    )

    return llm