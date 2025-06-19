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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

from langchain.chains import RetrievalQA

import torch

from accelerate import disk_offload

import base64

from sre_parse import Tokenizer

import accelerate

from langchain_openai import AzureChatOpenAI , AzureOpenAIEmbeddings

from dotenv import load_dotenv

import logging, verboselogs

from time import sleep

import os

from deepgram import (

DeepgramClient,

DeepgramClientOptions,

LiveTranscriptionEvents,

LiveOptions,

Microphone,

)

load_dotenv()

from transformers import BarkModel, BarkProcessor

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

txt = ""

arr = []

final_result = ""

try:

deepgram = DeepgramClient(os.getenv('api_deepgram'))

dg_connection = deepgram.listen.live.v("1")

def on_message(self, result, **kwargs):

global txt

sentence = result.channel.alternatives[0].transcript

if len(sentence) == 0:

return

# print(f"speaker: {sentence}")

txt += "/n"+sentence

arr.append(sentence)

print(txt)

def on_metadata(self, metadata, **kwargs):

print(f"/n/n{metadata}/n/n")

def on_speech_started(self, speech_started, **kwargs):

print(f"/n/n{speech_started}/n/n")

# Deepgram’s Utterances feature allows the chosen model to interact more naturally and effectively with

# speakers' spontaneous speech patterns

def on_utterance_end(self, utterance_end, **kwargs):

print(f"/n/n{utterance_end}/n/n")

def on_error(self, error, **kwargs):

print(f"/n/n{error}/n/n")

dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)

dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)

dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)

dg_connection.on(LiveTranscriptionEvents.Error, on_error)

options = LiveOptions(

model="nova-2",

punctuate=True,

language="en-US",

encoding="linear16",

channels=1,

sample_rate=16000,

# To get UtteranceEnd, the following must be set:

interim_results=True,

utterance_end_ms="1000",

vad_events=True,

)

dg_connection.start(options, addons=dict(myattr="hello"), test="hello")

# Open a microphone stream on the default input device

microphone = Microphone(dg_connection.send)

# start microphone

microphone.start()

# wait until finished

input("Press Enter to stop recording.../n/n")

# Wait for the microphone to close

microphone.finish()

# Indicate that we've finished

dg_connection.finish()

print("Finished")

# print(txt[-1])

# sleep(10)  # wait 30 seconds to see if there is any additional socket activity

# print("Really done!")

except Exception as e:

print(f"Could not open socket: {e}")

# return

# print("Final result:" + arr[-1])

query = arr[-1]

print(query)

retriever=db.as_retriever()

OPENAI_DEPLOYMENT_ENDPOINT =""

OPENAI_API_KEY = ""

OPENAI_DEPLOYMENT_NAME = ""

OPENAI_DEPLOYMENT_VERSION = ""

OPENAI_MODEL_NAME=""

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = ""

OPENAI_ADA_EMBEDDING_MODEL_NAME = ""

llm = AzureChatOpenAI(openai_api_key = OPENAI_API_KEY,

deployment_name = OPENAI_DEPLOYMENT_NAME,

model_name = OPENAI_MODEL_NAME,

api_version = OPENAI_DEPLOYMENT_VERSION,

azure_endpoint = OPENAI_DEPLOYMENT_ENDPOINT,

max_tokens=150

)

template = """Answer the question based only on the following context:

{context}

Question: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

model = llm

chain = (

{"context": retriever, "question": RunnablePassthrough()}

| prompt

| model

| StrOutputParser()

)

response = chain.invoke(query)

print(response)

import requests

# Define the API endpoint

url = "https://api.deepgram.com/v1/speak?model=aura-stella-en"

# Set your Deepgram API key

api_key = "f93d2941da5ddd5877d3b0a73ebbb612613d1ead"

# Define the headers

headers = {

"Authorization": f"Token {api_key}",

"Content-Type": "application/json"

}

# Define the payload

payload = {

"text": response

}

# Make the POST request

output = requests.post(url, headers=headers, json=payload)

# Check if the request was successful

if output.status_code == 200:

# Save the response content to a file

with open("audio_output.wav", "wb") as f:

f.write(output.content)

print("File saved successfully.")

audio_data = open("output_audio.wav", "rb").read()

Audio(audio_data, autoplay=True)

else:

print(f"Error: {output.status_code} - {output.text}")