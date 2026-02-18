from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = "vectorstore"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_and_split_pdf(pdf_path, chunk_size=500, chunk_overlap=50):
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks):

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save locally
    vectorstore.save_local(VECTOR_DB_PATH)

    return len(chunks)
def format_docs(docs):
    """Helper to join retrieved document chunks into one string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain():

    # 1. Load Vector Store
    # allow_dangerous_deserialization is required for .pkl files you created yourself
    vectorstore = FAISS.load_local(
        "vectorstore", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Use ChatPromptTemplate for Llama-3-Instruct
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that answers questions based on the uploaded document. Use the context provided to give a detailed answer.If the answer is not in the context, say \"I don't know.\""),
    ("user", "Document Context:\n{context}\n\nUser Question: {question}")
    ])

    # 3. Setup LLM with the Chat wrapper
    base_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        temperature=0.3,
        max_new_tokens=500
    )
    chat_model = ChatHuggingFace(llm=base_llm)

    # 4. The RAG Pipeline (LCEL)
    rag_chain = (
        {
            "context": retriever | format_docs, # Automatically retrieves AND formats
            "question": RunnablePassthrough()    # Passes the user question directly
        }
        | prompt 
        | chat_model 
        | StrOutputParser() # Ensures you get a clean string back
    )

    return rag_chain