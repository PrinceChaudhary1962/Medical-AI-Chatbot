import os
import io
import json
import math
import hashlib
from typing import List, Dict, Tuple

import numpy as np
import pdfplumber
import faiss
import tiktoken

# Optional: swap OpenAI for sentence-transformers if you prefer local embeddings.
# from sentence_transformers import SentenceTransformer


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def extract_pdf_text(file_bytes: bytes) -> List[Dict]:
    """
    Returns a list of dicts: {page_num, text}
    """
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Normalize whitespace
            text = " ".join(text.split())
            if text.strip():
                pages.append({"page_num": i, "text": text})
    return pages


def _tokenizer():
    # GPT-4o/-4 embeddings compatible tokenizer
    return tiktoken.get_encoding("cl100k_base")


def chunk_pages(pages: List[Dict], chunk_tokens: int = 600, overlap: int = 80) -> List[Dict]:
    """
    Sliding-window chunking across page texts with token overlap.
    Returns list of chunks with fields: {id, page_start, page_end, text}
    """
    enc = _tokenizer()
    chunks = []
    for p in pages:
        tokens = enc.encode(p["text"])
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            sub = enc.decode(tokens[start:end])
            chunks.append({
                "id": f"p{p['page_num']}_t{start}",
                "page_start": p["page_num"],
                "page_end": p["page_num"],
                "text": sub.strip()
            })
            if end == len(tokens):
                break
            start = max(0, end - overlap)
    # light merge for tiny trailing chunks: (optional)
    return chunks


def build_embeddings_openai(chunks: List[Dict], client, model: str = "text-embedding-3-small") -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vectors = []
    # Batch to stay under token limits
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=model, input=texts[i:i+batch_size])
        vectors.extend([np.array(d.embedding, dtype="float32") for d in resp.data])
    return np.vstack(vectors)


def create_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def search(query: str, client, chunks: List[Dict], index, emb_model: str = "text-embedding-3-small", top_k: int = 4):
    q = client.embeddings.create(model=emb_model, input=[query]).data[0].embedding
    q = np.array(q, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        ch = chunks[idx]
        results.append({
            "score": float(score),
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "text": ch["text"]
        })
    return results


def format_context_for_prompt(hits: List[Dict]) -> str:
    blocks = []
    for h in hits:
        pages = f"p. {h['page_start']}" if h['page_start'] == h['page_end'] else f"pp. {h['page_start']}-{h['page_end']}"
        blocks.append(f"[{pages}] {h['text']}")
    return "\n\n".join(blocks)


def avg_confidence(hits: List[Dict]) -> float:
    if not hits:
        return 0.0
    # map cosine ([-1,1]) normalized IP to [0,1] for display
    # scores are inner products of normalized vectors so they are cosine similarities
    sims = [max(0.0, min(1.0, (h["score"] + 1) / 2)) for h in hits]
    return float(sum(sims) / len(sims))
