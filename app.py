import streamlit as st
import streamlit as st
from langchain_community.vectorstores import DeepLake
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI , AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from st_audiorec import st_audiorec
from deepgram import Deepgram
from dotenv import load_dotenv
load_dotenv()
import requests
from io import BytesIO
import IPython.display as ipd
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader,PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import FAISS

OPENAI_DEPLOYMENT_ENDPOINT = os.getenv(OPENAI_DEPLOYMENT_ENDPOINT)
OPENAI_API_KEY = os.getenv(OPENAI_API_KEY)
OPENAI_DEPLOYMENT_NAME = os.getenv(OPENAI_DEPLOYMENT_NAME)
OPENAI_DEPLOYMENT_VERSION = os.getenv(OPENAI_DEPLOYMENT_VERSION)
OPENAI_MODEL_NAME=""
OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = ""
OPENAI_ADA_EMBEDDING_MODEL_NAME = ""
from langchain_community.vectorstores import Chroma 
from langchain_community.vectorstores import DeepLake
from langchain_community.vectorstores import deeplake
import os
persist_directory = "db"



# def ingestion():
#     for root, dirs, files in os.walk("docs"):
#         for file in files:
#             if file.endswith(".pdf"):
#                 print(file)
#                 loader = PDFMinerLoader(os.path.join(root, file))
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#         texts = text_splitter.split_documents(documents)
#     #create embeddings here
#         embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     #create vector store here
#         db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
#         db.add_documents(texts)






api_key = "api_deepgram"  # Replace with your Deepgram API key





# Set your Deepgram API key

# Deepgram API setup
with st.sidebar:
    
    docs_folder = "docs"
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    # loader = None  # Initialize loader to None before the loop
    # for root, dirs, files in os.walk("docs"):
    #     for file in files:
    #         if file.endswith(".pdf"):
    #             print(file)
    #             loader = PDFMinerLoader(os.path.join(root, file))

    # # Check if loader is still None after the loop
    # if loader is None:
    #     print("No PDF files found in the specified directory.")
        
        
        
        
    if uploaded_file is not None:
    # Save the uploaded file to the docs folder
        file_path = os.path.join(docs_folder , uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        # with BytesIO(uploaded_file.getvalue()) as pdf_data:
        loader = PDFMinerLoader(file_path)
        

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings here
    embeddings = AzureOpenAIEmbeddings(openai_api_key = OPENAI_API_KEY,
                                       deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type="azure",
                                chunk_size=1)

    # Create vector store here
    # db = DeepLake(dataset_path="./one_my_deeplake/", embedding=embeddings, overwrite=True)
    # db.add_documents(texts)
    db = FAISS.from_documents(texts, embeddings)

    #create embeddings here
    

    st.info("🚀 Sign up for a [Free API key](https://console.deepgram.com/signup)")
    deepgram_api_key = st.text_input(
        "🔐 Deepgram API Key",
        type="password",
        placeholder="Enter your Deepgram API key",
    )

    if deepgram_api_key == "":
        st.error("Please enter your Deepgram API key to continue")
        st.stop()

    deepgram = Deepgram(deepgram_api_key)

st.title("Voice Interaction with Large Language Model")

# Record audio from microphone
st.session_state["audio"] = st_audiorec()

if st.session_state["audio"]:
    st.audio(st.session_state["audio"])

    # Transcribe audio using Deepgram API
    if st.button("Ask a Question", type="primary"):
        st.spinner("Transcribing audio...")

        # Transcribe the audio using Deepgram API
        try:
            response = deepgram.transcription.sync_prerecorded(
                source={"buffer": st.session_state["audio"], "mimetype": "audio/wav"},
                model="nova-2",
                language="en",
            )
            transcribed_text = response["results"]["channels"][0]["alternatives"][0]["transcript"]

            st.write("🎙️ **You Asked:**")
            st.write(transcribed_text)

            query = transcribed_text
            # Query Large Language Model
            # (Replace the following line with your actual LLN processing logic) 
            OPENAI_DEPLOYMENT_ENDPOINT =OPENAI_DEPLOYMENT_ENDPOINT
            OPENAI_API_KEY = OPENAI_API_KEY
            OPENAI_DEPLOYMENT_NAME = OPENAI_DEPLOYMENT_NAME
            OPENAI_DEPLOYMENT_VERSION = OPENAI_DEPLOYMENT_VERSION
            OPENAI_MODEL_NAME=OPENAI_MODEL_NAME
 
# Streamlit UI

# User input for query

# Button to trigger the processing
    # Azure Chat OpenAI setup
            llm = AzureChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        model_name=OPENAI_MODEL_NAME,
        api_version=OPENAI_DEPLOYMENT_VERSION,
        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
        max_tokens=150
    )

    # Template for prompt
            template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
            prompt = ChatPromptTemplate.from_template(template)
            retriever = db.as_retriever()

    # Model setup
            model = llm

    # # Define the processing chain
            chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Invoke the chain with the user's question
            response = chain.invoke(query)
            output = response
            st.write(output)
            # st.write(type(response))
            # llm_response = "This is a placeholder response from the Large Language Model."

            # Convert LLN response to speech
            st.info("Converting LLN response to speech...")

            tts_payload = {"text": response}
            tts_response = requests.post(
                url="https://api.deepgram.com/v1/speak?model=aura-stella-en",
                headers={"Authorization": f"Token {api_key}"},
                json=tts_payload,
                verify=False,
            )

            # Check if the request was successful
            if tts_response.status_code == 200:
                # Display the generated audio
                audio_bytes = BytesIO(tts_response.content)
                st.audio(audio_bytes, format="audio/mp3", start_time=0)

                # Play the audio automatically
                ipd.Audio(tts_response.content)
            else:
                st.error(f"Error during Text-to-Speech: {tts_response.status_code}")

        except Exception as e:
            st.error(f"Error during transcription: {e}")
