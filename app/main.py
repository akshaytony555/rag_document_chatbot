from fastapi import FastAPI,UploadFile,File
from app.rag import load_and_split_pdf
import os
import shutil

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to the RAG Document Chatbot API!"}

@app.post("/chat")
def upload_document_and_split(file:UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = load_and_split_pdf(file_path)
    return  {  "filename": file.filename,  "num_chunks": len(chunks)}
   
