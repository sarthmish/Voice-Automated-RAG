from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader,PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma 
from langchain_community.vectorstores import DeepLake
import os
persist_directory = "db"

for root, dirs, files in os.walk("docs"):
    for file in files:
        if file.endswith(".pdf"):
            print(file)
            loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
    db.add_documents(texts)