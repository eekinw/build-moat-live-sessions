import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from . import indexer


SYSTEM_PROMPT = """
You are a grounded knowledge base Q&A assistant.
  
Rules:
- Answer only from the CONTEXT in the user message.
- Cite claims with the exact source IDs shown in [Source: ...].
- Do not cite sources that are not present in the CONTEXT.
- If the CONTEXT does not contain enough evidence, say: "I cannot confirm from the knowledge base."
- Do not guess, infer from outside knowledge, or use general world knowledge.
"""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            timeout=20,
            max_retries=1,
        )
    return _llm

# Build the prompt from retrieved vector chunks.
#
# Design decision: Give the LLM enough context without flooding it.
#
# Hints:
# 1. Include [Source: filename#heading] before each chunk.
# 2. Include retrieval distance or score only for debugging.
# 3. Use top-k chunks passed into this function.
# 4. Place CONTEXT before QUESTION.
def build_prompt(query: str, ranked_chunks: list) -> str:
    context_blocks = []
    for doc, _ in ranked_chunks:
        source = doc.metadata.get("source", "unknown") # unknown as fallback
        context_blocks.append(
            "\n".join([
                f"[Source: {source}]",
                doc.page_content,
            ])
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
