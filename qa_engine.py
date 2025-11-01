import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

load_dotenv()

VECTOR_DIR = 'vectorstore'

def load_qa_chain():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})
    llm = ChatOpenAI(model='gpt-4-turbo', temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

if __name__ == '__main__':
    qa = load_qa_chain()
    while True:
        q = input('Ask a question (or type exit): ')
        if q.strip().lower() in ('exit','quit'):
            break
        print('\nAnswer:\n', qa.run(q))
