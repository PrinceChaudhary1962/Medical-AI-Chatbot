# Medical RAG Chatbot (Streamlit + FAISS + OpenAI)

A ready-to-run Retrieval Augmented Generation (RAG) chatbot that answers questions **only** from a medical PDF you provide. It uses:
- **pdfplumber** for PDF text extraction
- **FAISS** for vector search
- **OpenAI** for embeddings and chat completion
- **Streamlit** for a simple, clean UI

> ⚠️ **Safety**: This app is for **educational purposes only**. It does **not** provide medical advice, diagnosis, or treatment. Always consult a licensed clinician for medical concerns.

## Quickstart (Local)

1. **Install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your key**
   - Either create `.streamlit/secrets.toml` with:
     ```toml
     OPENAI_API_KEY = "sk-..."
     ```
   - Or export an environment variable:
     ```bash
     export OPENAI_API_KEY="sk-..."
     ```

3. **Run**
   ```bash
   streamlit run app.py
   ```

4. **Upload your medical PDF** in the sidebar and start chatting.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. Create a new app on Streamlit Cloud and point it to `app.py`.
3. In **App settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Deploy. Upload your PDF in the running app.

## Notes

- Chunking/overlap, top-k retrieval, temperature and model choices are adjustable in the sidebar.
- Answers include inline citations like `[p. 42]` and a **Show sources** section.
- The FAISS index is built per-file and cached; re-uploads rebuild the index.
- Want to use an open-source stack? See the comments in `utils.py` for switching to `sentence-transformers` and a local LLM.
