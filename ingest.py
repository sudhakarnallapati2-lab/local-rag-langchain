import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

DATA_DIR = 'data'
VECTOR_DIR = 'vectorstore'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

def load_documents():
    docs = []
    # PDF loader
    pdf_loader = DirectoryLoader(DATA_DIR, glob='**/*.pdf', loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())
    # TXT loader as fallback
    txt_loader = DirectoryLoader(DATA_DIR, glob='**/*.txt', loader_cls=TextLoader)
    docs.extend(txt_loader.load())
    print(f'Loaded {len(docs)} documents.')
    return docs

def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f'Split into {len(chunks)} chunks.')
    return chunks

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)
    print(f'FAISS index saved to {VECTOR_DIR}/')

if __name__ == '__main__':
    documents = load_documents()
    chunks = create_chunks(documents)
    build_vectorstore(chunks)
