# Streamlit RAG app with upload, conversational memory, multi-file selection, and downloads.
import os, io
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
try:
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

from langchain.prompts import PromptTemplate
from fpdf import FPDF

load_dotenv()

st.set_page_config(page_title='RAG Multi-file Chat', page_icon='🤖')

USE_OLLAMA = os.getenv('OLLAMA','').lower() in ('1','true','yes') or os.getenv('USE_OLLAMA','').lower() in ('1','true','yes')

st.title('📚 Local RAG — Multi-file aware Chat')

DATA_DIR = 'data'
VECTOR_DIR = 'vectorstore'
os.makedirs(DATA_DIR, exist_ok=True)

# Session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Helper: process uploaded files and (re)build main FAISS index
def process_and_build(files):
    all_docs = []
    for f in files:
        path = os.path.join(DATA_DIR, f.name)
        with open(path, 'wb') as out:
            out.write(f.read())
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    if USE_OLLAMA and HAS_OLLAMA:
                        embeddings = OllamaEmbeddings(model='nomic-embed-text')
                    else:
                        embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)
    return db

# Upload UI
uploaded = st.file_uploader('Upload PDFs', type=['pdf'], accept_multiple_files=True)
if uploaded:
    st.session_state.vector_db = process_and_build(uploaded)
    st.success('FAISS index created from uploaded files.')

# If index exists on disk, load it
if os.path.exists(VECTOR_DIR) and st.session_state.vector_db is None:
    if USE_OLLAMA and HAS_OLLAMA:
                        embeddings = OllamaEmbeddings(model='nomic-embed-text')
                    else:
                        embeddings = OpenAIEmbeddings()
    try:
        st.session_state.vector_db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        st.success('Loaded existing FAISS index.')
    except Exception as e:
        st.warning('Failed to load FAISS index: ' + str(e))

# List available files in data/
available_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
st.sidebar.markdown('### Files in data/')
for f in available_files:
    st.sidebar.write(f)

# Multi-file selection (if user wants to restrict search)
selected_files = st.multiselect('Select files to include in search (leave empty for all)', options=available_files)

# Build or get QA chain
if st.session_state.vector_db is not None:
    # Conversational chain (uses memory)
    if USE_OLLAMA and HAS_OLLAMA:
            llm = Ollama(model='llama3')
        else:
            llm = ChatOpenAI(model='gpt-4-turbo', temperature=0)
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={'k': 3})
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=st.session_state.memory, return_source_documents=True
    )

# Display chat history
if st.session_state.messages:
    st.markdown('### Chat History')
    for m in st.session_state.messages:
        st.markdown(f'**You:** {m["question"]}')
        st.markdown(f'**AI:** {m["answer"]}')
        if m.get('sources'):
            with st.expander('Sources'):
                for s in m['sources']:
                    st.write('- ' + s)

# Query box
if st.session_state.qa_chain is not None:
    query = st.text_input('Ask a question (conversational):')
    if query:
        with st.spinner('Thinking...'):
            # If user selected files, create a temporary index from those files
            if selected_files:
                # load docs only from selected files and create a temp FAISS
                temp_docs = []
                for fname in selected_files:
                    p = os.path.join(DATA_DIR, fname)
                    if os.path.exists(p):
                        temp_docs.extend(PyPDFLoader(p).load())
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                temp_chunks = splitter.split_documents(temp_docs)
                if USE_OLLAMA and HAS_OLLAMA:
                        embeddings = OllamaEmbeddings(model='nomic-embed-text')
                    else:
                        embeddings = OpenAIEmbeddings()
                temp_db = FAISS.from_documents(temp_chunks, embeddings)
                temp_retriever = temp_db.as_retriever(search_kwargs={'k':3})
                # use a temporary conversational chain against temp_retriever
                temp_chain = ConversationalRetrievalChain.from_llm(
                    llm=(Ollama(model='llama3') if (USE_OLLAMA and HAS_OLLAMA) else ChatOpenAI(model='gpt-4-turbo', temperature=0)),
                    retriever=temp_retriever,
                    memory=st.session_state.memory,
                    return_source_documents=True
                )
                res = temp_chain({'question': query})
            else:
                # Use main chain on all documents
                res = st.session_state.qa_chain({'question': query})

            answer = res.get('answer') or res.get('result')
            srcs = []
            for d in res.get('source_documents', []):
                src = os.path.basename(d.metadata.get('source', 'Unknown'))
                page = d.metadata.get('page', 'N/A')
                srcs.append(f"{src} (Page {page})")
            st.session_state.messages.append({'question': query, 'answer': answer, 'sources': srcs})

            st.markdown('### Answer')
            st.write(answer)
            if srcs:
                st.markdown('### Sources')
                for s in srcs:
                    st.write('- ' + s)

else:
    st.info('Upload PDFs or build the index first.')

# Export utilities
def export_md(messages):
    md = '# Chat History\n\n'
    for i,m in enumerate(messages,1):
        md += f'### Q{i}: {m["question"]}\n'
        md += f'**A{i}:** {m["answer"]}\n\n'
        if m.get('sources'):
            md += '**Sources:**\n'
            for s in m['sources']:
                md += f'- {s}\n'
        md += '\n---\n'
    return md

def export_pdf(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.multi_cell(0,10,'Chat History\n\n')
    for i,m in enumerate(messages,1):
        pdf.multi_cell(0,10,f'Q{i}: {m["question"]}\nA{i}: {m["answer"]}\n')
        if m.get('sources'):
            pdf.multi_cell(0,10,'Sources:')
            for s in m['sources']:
                pdf.multi_cell(0,10,'- ' + s)
        pdf.multi_cell(0,10,'\n')
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

if st.session_state.messages:
    st.sidebar.markdown('### Download history')
    st.sidebar.download_button('Download Markdown', data=export_md(st.session_state.messages), file_name='chat_history.md', mime='text/markdown')
    st.sidebar.download_button('Download PDF', data=export_pdf(st.session_state.messages), file_name='chat_history.pdf', mime='application/pdf')
