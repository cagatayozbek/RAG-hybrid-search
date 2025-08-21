# RAG-hybrid-search

# RAG-hybrid-search

## Project Description

This project is a **Retrieval-Augmented Generation (RAG)** system built with LangChain and LangGraph.  
The main goal is to experiment with **hybrid search** by combining **BM25 (sparse retrieval)** and **FAISS (dense retrieval)** into an **Ensemble Retriever**, enabling more robust document retrieval.  
We also integrated the retrievers with an LLM, explored **tool-based retrieval**, and experimented with **memory persistence** for conversational experiences.

---

## What We Have Done So Far

- Loaded and split documents into chunks for indexing.
- Built **hybrid search** using:
  - BM25Retriever (sparse)
  - FAISS Retriever (dense)
  - EnsembleRetriever (weighted combination)
- Connected the hybrid retriever to an LLM for **RAG-style Q&A**.
- Tested **memory with LangGraph checkpointers** to keep conversational history across turns.

---

## Tech Stack

- **Python 3.10+**
- **LangChain** (retrievers, LLM wrappers, embeddings)
- **LangGraph** (graph-based RAG workflows, memory)
- **FAISS** (vector store for dense retrieval)
- **BM25 Retriever** (sparse retrieval)
- \*\*Gemini 2.0 flash (generation)

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/RAG-hybrid-search.git
   cd RAG-hybrid-search
   ```

2. **Create a virtual environment**

   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set environment variables**  
   Create a `.env` file in the project root:

   ```bash
    GOOGLE_API_KEY=your_api_key_here
    COHERE_API_KEY=your_api_key_here
    HF_TOKEN=your_api_key_here
   ```

5. **Run example scripts**

   - Launch Gradio UI for chat:

     ```bash
     python chat_ui.py
     ```

   - Alternatively, you can serve the application via API using `uvicorn`:
     ```bash
     uvicorn main:app --reload --port 8000
     ```

---

## Next Steps

    - Improve the frontend: Instead of the basic Gradio UI, build a more feature-rich frontend (e.g., with React, Next.js) to provide a smoother chat experience, better history management, and customization options. So frontend file is empty now.
