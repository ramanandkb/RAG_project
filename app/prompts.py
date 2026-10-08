from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are a helpful customer support assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I cannot find the answer in the provided knowledge base."

Do not invent information.

Context:
{context}
"""


def get_prompt():

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])