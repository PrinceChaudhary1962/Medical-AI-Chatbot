import os
import io
import time
import json
import streamlit as st
import numpy as np

from openai import OpenAI
from utils import (
    get_file_hash,
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
Ask questions and get answers **grounded** in your uploaded medical ebook.

> ⚠️ **Educational use only.** Not medical advice. Always seek the guidance of a licensed clinician.
""")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o"], index=0)
    emb_model = st.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    chunk_tokens = st.slider("Chunk size (tokens)", 300, 1200, 600, 50)
    overlap = st.slider("Chunk overlap (tokens)", 0, 200, 80, 10)
    top_k = st.slider("Top-K passages", 1, 10, 4, 1)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    uploaded = st.file_uploader("Upload a medical PDF", type=["pdf"])
    st.caption("Upload your ebook to build a searchable index.")

# Resolve API key
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
client = OpenAI(api_key=api_key) if api_key else None

# Cache index per file
@st.cache_resource(show_spinner=True)
def build_index(file_bytes: bytes, chunk_tokens: int, overlap: int, emb_model: str):
    pages = extract_pdf_text(file_bytes)
    if not pages:
        raise ValueError("No extractable text found in the PDF.")
    chunks = chunk_pages(pages, chunk_tokens=chunk_tokens, overlap=overlap)
    embs = build_embeddings_openai(chunks, client, model=emb_model)
    index = create_faiss_index(embs)
    meta = {"pages": pages, "chunks": chunks}
    return index, meta

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload your medical PDF in the sidebar and ask me anything about its contents."}
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

index = None
meta = None

if uploaded is not None:
    file_bytes = uploaded.read()
    with st.status("Building index...", expanded=False):
        try:
            index, meta = build_index(file_bytes, chunk_tokens, overlap, emb_model)
            st.success("Index ready.")
        except Exception as e:
            st.error(f"Failed to index PDF: {e}")
else:
    st.info("Please upload a medical PDF to start.")

# Chat input
prompt = st.chat_input("Ask a question about the ebook...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if index is None:
            st.markdown("Please upload a PDF first so I can search it.")
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
            badge = f"**Retrieval confidence:** {conf:.2f}"

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

            # Always append disclaimer
            st.info("⚠️ Educational use only. Not medical advice. Consult a licensed clinician.")
            st.session_state.messages.append({"role": "assistant", "content": answer})
