import os
import streamlit as st
import numpy as np

from openai import OpenAI
from utils import (
    extract_pdf_text,
    chunk_pages,
    build_embeddings_openai,
    create_faiss_index,
    search,
    format_context_for_prompt,
    avg_confidence,
)

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺", layout="wide")

st.markdown("""
# 🩺 Medical RAG Chatbot
Ask questions and get answers **grounded** in the provided medical ebook.

> ⚠️ **Educational use only.** Not medical advice. Always seek the guidance of a licensed clinician.
""")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o"], index=0)
    emb_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    chunk_tokens = st.slider("Chunk size (tokens)", 300, 1200, 600, 50)
    overlap = st.slider("Chunk overlap (tokens)", 0, 200, 80, 10)
    top_k = st.slider("Top-K passages", 1, 10, 4, 1)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.caption("Default PDF from repo will be used (no upload needed).")

# API key
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
client = OpenAI(api_key=api_key) if api_key else None

# Default PDF path
DEFAULT_PDF = "medical copy_compressed.pdf"

@st.cache_resource(show_spinner=True)
def build_index_from_path(pdf_path: str, chunk_tokens: int, overlap: int, emb_model: str):
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    pages = extract_pdf_text(file_bytes)
    if not pages:
        raise ValueError("No extractable text found in the PDF.")
    chunks = chunk_pages(pages, chunk_tokens=chunk_tokens, overlap=overlap)
    embs = build_embeddings_openai(chunks, client, model=emb_model)
    index = create_faiss_index(embs)
    meta = {"pages": pages, "chunks": chunks}
    return index, meta

index, meta = None, None
if os.path.exists(DEFAULT_PDF):
    with st.status("Building index from default PDF...", expanded=False):
        try:
            index, meta = build_index_from_path(DEFAULT_PDF, chunk_tokens, overlap, emb_model)
            st.success("Index ready (default PDF loaded).")
        except Exception as e:
            st.error(f"Failed to index PDF: {e}")
else:
    st.error(f"Default PDF not found at {DEFAULT_PDF}. Please add it to the repo.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! The medical PDF is preloaded. Ask me anything about its contents."}
    ]

# Display chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def make_system_prompt():
    return (
        "You are a medical tutoring assistant restricted to the provided context. "
        "Answer ONLY using the context. If the answer is not present, say you don't know. "
        "Cite sources inline with page numbers like [p. 12]. "
        "Style: clear, concise, professional.\n\n"
        "CRITICAL SAFETY: This is for educational purposes only and is not medical advice. "
        "Do not diagnose, prescribe, or suggest treatments. Urge users to consult a licensed clinician for medical concerns."
    )

def assemble_messages(user_query: str, context_blocks: str):
    system = make_system_prompt()
    user = (
        f"Question:\n{user_query}\n\n"
        f"Context (excerpts from the ebook):\n{context_blocks}\n\n"
        "When you cite, use the nearest page number like [p. 42]."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

# Chat input
prompt = st.chat_input("Ask a question about the ebook...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if index is None:
            st.markdown("PDF not indexed yet.")
        elif client is None:
            st.markdown("OpenAI API key not set. Add it in secrets or environment.")
        else:
            hits = search(prompt, client, meta["chunks"], index, emb_model=emb_model, top_k=top_k)
            context_blocks = format_context_for_prompt(hits)
            messages = assemble_messages(prompt, context_blocks)

            with st.spinner("Thinking..."):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
                    answer = resp.choices[0].message.content
                except Exception as e:
                    answer = f"Error from model: {e}"

            conf = avg_confidence(hits)
            badge = f"**Retrieval confidence:** {conf:.2f}**"

            st.markdown(answer)
            st.markdown(badge)

            with st.expander("Show sources"):
                if hits:
                    for i, h in enumerate(hits, start=1):
                        pages = f"[p. {h['page_start']}]" if h["page_start"] == h["page_end"] else f"[pp. {h['page_start']}-{h['page_end']}]"
                        snippet = h["text"][:400].strip().replace("\n", " ")
                        st.markdown(f"**{i}. {pages} (score {h['score']:.3f})**\n\n> {snippet}…")
                else:
                    st.write("No sources retrieved.")

            st.info("⚠️ Educational use only. Not medical advice. Consult a licensed clinician.")
            st.session_state.messages.append({"role": "assistant", "content": answer})
