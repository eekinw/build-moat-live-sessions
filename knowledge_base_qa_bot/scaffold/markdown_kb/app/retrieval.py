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


def build_prompt(query: str, ranked_sections: list) -> str:
    # TODO: Build the prompt from top-ranked Markdown sections.
    #
    # Design decision: Put raw Markdown sections into CONTEXT with citations.
    #
    # Hints:
    # 1. Include [Source: filename#heading] before each section.
    # 2. Include heading_path so the model sees the document structure.
    # 3. Include only top sections passed into this function.
    # 4. Place CONTEXT before QUESTION.
    context_blocks = []
    for section, _ in ranked_sections:
        heading_path = " > ".join(section.heading_path)
        context_blocks.append(
            "\n".join([
                f"[Source: {section.id}]",
                f"Heading: {heading_path}",
                section.content,
            ])
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

def query(question: str) -> dict:
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=3)
    if not ranked_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
