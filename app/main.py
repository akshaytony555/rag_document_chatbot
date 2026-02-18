from fastapi import FastAPI,UploadFile,File,Form
from app.rag import load_and_split_pdf,create_vector_store,get_qa_chain
import os
import shutil

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to the RAG Document Chatbot API!"}

@app.post("/upload/")
def upload_document_and_split(file:UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = load_and_split_pdf(file_path)
    num_chunks = create_vector_store(chunks)
    return  {  "filename": file.filename,  "num_chunks": num_chunks,"message":"Vector store created successfully!"}
   
@app.post("/ask/")
def ask_question(question: str = Form(...)):
    chain = get_qa_chain()
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer
    }