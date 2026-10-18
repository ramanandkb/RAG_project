
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from app.config import PINECONE_INDEX_NAME
from app.embeddings import get_embeddings
from app.llm import get_llm

load_dotenv()

NAMESPACE = os.getenv("PINECONE_NAMESPACE", "worldbank-docs")

SYSTEM_PROMPT = """
You are a World Bank Research Assistant.

Answer using ONLY the provided context.

Instructions:
1. Give a direct, natural-language answer.
2. Do not display raw chunks or similarity scores.
3. Never invent facts or use outside knowledge.
4. If the answer is unavailable, say:
   "I cannot find the answer in the provided knowledge base."
5. Cite the source filename and page number when available.
6. If the context is insufficient, explain the limitation.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

vector_store = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, llm

    # Initialize once when the server starts
    embeddings = get_embeddings()
    llm = get_llm()

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE
    )

    yield


app = FastAPI(
    title="World Bank Research Assistant",
    description="RAG API using Pinecone and Groq",
    version="1.0.0",
    lifespan=lifespan
)

# Restrict this to your frontend domain when you deploy it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {"message": "World Bank Research Assistant API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    try:
        results = vector_store.similarity_search(
            query=question,
            k=4
        )

        if not results:
            return {
                "question": question,
                "answer": (
                    "I cannot find the answer in the provided "
                    "knowledge base."
                ),
                "sources": []
            }

        context_parts = []
        sources = []

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

            item = {"source": source, "page": page}
            if item not in sources:
                sources.append(item)

        context = "\n\n".join(context_parts)

        messages = prompt.format_messages(
            question=question,
            context=context
        )

        response = llm.invoke(messages)

        return {
            "question": question,
            "answer": response.content,
            "sources": sources
        }

    except Exception:
        # Avoid exposing API keys or internal error details.
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question."
        )
