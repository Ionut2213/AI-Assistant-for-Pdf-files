import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama


# Config logging
logging.basicConfig(level=logging.INFO)




# Constants

DOC_PATH = "" # your pdf file location
MODEL_NAME = "llama3.2" # specify your model version
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag-model"
PERSISTENT_DIRECTORY = "chroma_db"





def load_pdf_file_in_the_project(doc_path):
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("Your PDF file was loaded successfully!")
        return data
    else:
        logging.error(f"Your PDF file wasn't found at pat {doc_path}")
        st.error("PDF file wasn't found")
        return None


def split_the_pdf_in_small_parts(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap= 300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Document was successfully split in small parts")
    return chunks





@st.cache_resource


def load_create_vector_db():
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSISTENT_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSISTENT_DIRECTORY
        )
        logging.info("Loaded your existing vector database")
    
    else:
        data = load_pdf_file_in_the_project(DOC_PATH)
        if data is None:
            return None
        
        chunks = split_the_pdf_in_small_parts(data)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSISTENT_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database was created")
    return vector_db



def create_multi_query_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=""" You are an AI language model assistant. Your task is to generate five different versions
        of the given user question to retrive relevant documents from a vector database. Provide these alternative questions separated
        by newline.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created successfully.")
    return retriever





def create_chain(retriever, llm):
    template = """" Answer the question based ONLY on the following context:
    {context}
    Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created with preserved syntax")
    return chain


def main():
    st.title("Personal Documents Assistant")

    user_input = st.text_input("Please enter your question:", "")

    if user_input:
        with st.spinner("Your response is generated....."):
            try:
                llm = ChatOllama(model = MODEL_NAME)

                vector_db = load_create_vector_db()
                if vector_db is None:
                    st.error("ERROR! Failed to load or create the vector database.")
                    return
                retriever = create_multi_query_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)
                response = chain.invoke(input = user_input)
                st.markdown("**Personal Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error was occurred: {str(e)}")

    else:
        st.info("Please enter a question to get started")


if __name__ == "__main__"():
    main()