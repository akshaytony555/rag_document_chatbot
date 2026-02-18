# RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions based on their content.

## Features
- Upload and process PDF files
- Automatic text chunking
- Embedding generation using Sentence Transformers
- Vector search using FAISS
- Context-aware question answering
- FastAPI-based backend

## Tech Stack
- Python
- FastAPI
- LangChain
- Hugging Face Embeddings
- FAISS
- Sentence Transformers

## How It Works
1. User uploads a PDF.
2. Text is extracted and split into chunks.
3. Chunks are converted into embeddings.
4. Embeddings are stored in a FAISS vector database.
5. User asks a question.
6. Relevant chunks are retrieved.
7. LLM generates an answer from the context.

## Installation
```bash
git clone https://github.com/akshaytony555/rag_document_chatbot.git
cd rag_document_chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
