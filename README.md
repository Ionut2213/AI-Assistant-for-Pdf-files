This application is an intelligent assitant for PDF documents, build using RAG(Retrival-Augumented Generation) tehnology.
The user can ask natural language questions, and the app will respond based on the content of a PDF file.


KEY FEATURES

1) Loads a PDF file(automatically, from a predefined location)
2) Split the document into logical chunks
3) Vectorizes the text using local embeddings(using Ollama)
4) Semantically searches for relevant fragments to the user's questions
5) Generates intelligent answer using local LLM(llama3.2)
6 User friendly interface build with Streamlit



TEHNOLGIES USED 
1) Python(Main Programming Language)
2) Streamlit(Rapid UI for building interactive web app)
3) Langchain(Framework for RAG and LLM integration)
4) Ollama(Local execution of LLMs and embeddding models)
5) ChromaDB(Local vector store for semantic search)
6) Unstructured(Extract text from unstructured PDF Documents)
7) RecursiveCharacterTextSplitter(Split the documents into chunks for embedding)
8) Llama3.2(Local LLM used for generationg answers)



NOTE

To make sure this app works you need to have OLLAMA installed


FUTURE FEATURES
1) Upload PDF files direct from UI
2) Option to save the istoric