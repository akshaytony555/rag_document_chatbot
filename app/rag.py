from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "vectorstore"

def load_and_split_pdf(pdf_path, chunk_size=500, chunk_overlap=50):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks):
     # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save locally
    vectorstore.save_local(VECTOR_DB_PATH)

    return len(chunks)