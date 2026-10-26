# DocuMind — Stylish Streamlit RAG Demo

Run from the folder containing `app.py`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Ensure Ollama is running and the models are installed:

```powershell
ollama pull llama3.2
ollama pull embeddinggemma
ollama list
```

Supports PDF, TXT, and DOCX. Scanned PDFs need OCR. ChromaDB uses in-memory storage in this demo, so the index resets when Streamlit restarts.




