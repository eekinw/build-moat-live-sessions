import json
import os
import re
import shutil
from pathlib import Path
import math

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import hashlib
from langchain_core.embeddings import Embeddings


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "local-hash-embeddings-v1"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
SYNONYMS = {
    "back": ["refund"],
    "how": ["timeline"],
    "long": ["timeline"],
    "money": ["refund"],
    "receive": ["processed"],
    "received": ["processed"],
    "take": ["processed", "timeline"],
    "takes": ["processed", "timeline"],
    "will": ["timeline"],
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80, # last 80 chars of each chunk are repeated at the start of the next, so facts at boundaries aren't lost
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0

class HashEmbeddings(Embeddings):
    """Small deterministic embedding model for local FAISS retrieval."""

    dimension = 256

    def _vectorize(self, text: str) -> list[float]:
        tokens = []
        for token in TOKEN_RE.findall(text.lower()):
            tokens.append(token)
            tokens.extend(SYNONYMS.get(token, []))

        vector = [0.0] * self.dimension

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 16, 2):
                bucket = int.from_bytes(digest[offset : offset + 2], "big") % self.dimension
                vector[bucket] += 1.0 if offset < 8 else 0.5

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)

# Configure chunking parameters for traditional RAG.
#
# Design decision: Balance semantic recall against context noise.
#
# Hints:
# 1. chunk_size around 500 chars is a reasonable prototype default.
# 2. chunk_overlap helps avoid cutting facts at boundaries.
# 3. separators should prefer Markdown structure before individual words.



def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HashEmbeddings()
    return _embeddings

# Load Markdown into source-citable Document records.
# Design decision: Preserve filename#heading metadata before chunking.

# Hints:
# 1. Use HEADING_RE to split by Markdown headings.
# 2. Put heading_path and content into page_content.
# 3. Store source metadata like "refund_policy.md#refund-timeline".
def load_markdown_sections(path: Path) -> list[Document]:
    documents = []
    current_heading = "Introduction"
    current_heading_path = path.name
    current_lines = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": f"{path.name}#{slugify(current_heading)}",
                        "heading": current_heading_path,
                    }
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(2).strip()
            current_heading_path = f"{path.name} > {current_heading}"
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return documents


# Build a FAISS vector index from docs/*.md.
#
# Hints:
# 1. Load all Markdown files from docs_dir.
# 2. Convert each heading section to a Document.
# 3. Split documents into chunks with splitter.split_documents().
# 4. Create FAISS.from_documents(chunks, get_embeddings()).
# 5. Save the FAISS index to .kb/faiss_index/.
# 6. Return (files_indexed, chunks_indexed).
def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed
    vectorstore = None
    files_indexed = 0
    sections_indexed = 0

    all_documents = []
    md_files = sorted(docs_dir.glob("*.md"))
    for path in md_files:
        all_documents.extend(load_markdown_sections(path))
    files_indexed = len(md_files)

    chunks = splitter.split_documents(all_documents)
    sections_indexed = len(chunks)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    save_vector_index()
    return files_indexed, sections_indexed

# Persist the FAISS index so restart does not require re-embedding.
#
# Hints:
# 1. Return early if vectorstore is None.
# 2. Clear stale persisted files with shutil.rmtree(...) if the new index is empty.
# 3. Use vectorstore.save_local(str(index_dir)).
# 4. Write metadata.json with embedding_model, files_indexed, and sections_indexed.
# 5. json.dumps(..., indent=2) makes the metadata easy to inspect.
def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        return
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

# Load .kb/faiss_index/ on server startup if it exists.
#
# Hints:
# 1. Check for index.faiss and index.pkl.
# 2. Read metadata.json and verify embedding_model still matches.
# 3. Use FAISS.load_local(..., allow_dangerous_deserialization=True).
# 4. Only use dangerous deserialization for indexes created by this local app.
def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed
    if not (index_dir / "index.faiss").exists():
        return 0, 0
    metadata_path = index_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError("Embedding model mismatch — rebuild the index")
    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    files_indexed = metadata.get("files_indexed", 0)
    sections_indexed = metadata.get("sections_indexed", 0)
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
