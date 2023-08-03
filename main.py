import os
from xml.dom.minidom import Document
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import langchain
langchain.debug=False

from langchain.prompts import PromptTemplate

if __name__ == "__main__":
    print("hello")
    
    pdf_path="2305.15334_gorilla.pdf"
    loader= PyPDFLoader(file_path=pdf_path)
    documents=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs=text_splitter.split_documents(documents=documents)
    
    embeddings=OpenAIEmbeddings()

    vectorstore=FAISS.from_documents(documents=docs,embedding=embeddings)
    vectorstore.save_local("faiss_index_gorilla")

    vectorstore=FAISS.load_local(folder_path="faiss_index_gorilla",embeddings=embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0, verbose=True), chain_type="stuff", retriever=vectorstore.as_retriever())
    res=qa.run("como fue entrenado gorilla?")
    print(res)


   
