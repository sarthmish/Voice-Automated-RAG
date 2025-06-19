# Voice-Automated-RAG

## 📌 Introduction

Retrieval-Augmented Generation (RAG) is a cutting-edge approach that combines document retrieval and generative language modeling to deliver accurate, context-aware responses. Our project enhances this paradigm by integrating **voice interaction**, enabling users to interact with documents using **spoken queries** and receive **natural-sounding voice responses**.

This system leverages:

- **STT (Speech-to-Text)** using Deepgram’s **Nova-2**
- **TTS (Text-to-Speech)** using Deepgram’s **Aura**
- **Vector databases** for retrieval
- **OpenAI GPT-3.5** as the Large Language Model (LLM)
- A **Streamlit** interface for seamless user interaction

## 🎯 Objective

- ✅ Develop a voice-driven interface to access document content
- ✅ Accurately convert user speech into text
- ✅ Retrieve relevant information using vector-based search
- ✅ Generate contextually appropriate responses via LLM
- ✅ Convert responses into clear, synthesized voice output
- ✅ Create an accessible, hands-free user experience

## 🧭 Scope

- Speech input via microphone
- Use of **Deepgram Nova-2** for speech recognition
- Vector-based document search using **FAISS** / **DeepLake**
- Response generation with **GPT-3.5 Turbo**
- Voice output using **Deepgram Aura**
- Multi-language support
- Intuitive UI using **Streamlit**
- Modular design for easy component swaps (e.g. different retrievers, LLMs)

## 📦 Key Deliverables

- 🔊 Voice-to-text to query system
- 📄 Document embedding and retrieval pipeline
- 🧠 Context-aware response generation
- 🗣️ Voice response synthesis
- 🖥️ Streamlit interface for user interaction

---

## 🧩 Problem Statement

Most traditional information retrieval systems require manual input and are not optimized for natural conversation. They fail to:

- Handle **voice queries**
- Understand **natural language**
- Deliver **real-time**, accurate, and **spoken responses**

This project solves that with a fully **automated, voice-driven RAG pipeline**.

---

## 💡 Approach

A seamless voice-based pipeline combining:

- 🎤 **Speech-to-Text** (Deepgram Nova-2)
- 🔍 **Vector-based retrieval** (Sentence Transformers + FAISS/DeepLake)
- 🧠 **LLM generation** (GPT-3.5 Turbo)
- 🔊 **Text-to-Speech** (Deepgram Aura)

---

## ⚙️ Workflow

### 1. Voice Input
User speaks into the microphone → captured by Deepgram Nova-2 (STT).

### 2. Query Processing
Text query is embedded using Sentence Transformers → relevant chunks fetched from FAISS/DeepLake.

### 3. Response Generation
Query + retrieved context → passed to GPT-3.5 → generates a coherent answer.

### 4. Voice Output
Generated answer → converted to voice using Deepgram Aura (TTS).

### 5. UI
Streamlit app enables real-time interaction and audio playback.

---

## 🧠 Use Case: Customer Support

Users can verbally ask questions about company policies, SOPs, or documentation. The system processes the query and responds instantly with voice—eliminating the need for human support agents and reducing query resolution time.

---

## 🛠 Technology Stack

| Component        | Tool / Library                  |
|------------------|----------------------------------|
| Programming Lang | Python                           |
| Embeddings       | Sentence Transformers            |
| Vector DB        | FAISS, DeepLake                  |
| STT              | Deepgram Nova-2                  |
| LLM              | OpenAI GPT-3.5 Turbo             |
| TTS              | Deepgram Aura                    |
| UI               | Streamlit                        |
| Frameworks       | Hugging Face, Langchain, Azure OpenAI |

---

## 🚧 Limitations & Future Work

### Current Limitation:
- Lacks memory for follow-up or contextual chaining in conversation.

### Upcoming Improvements:
- Implement **buffer memory** and **conversational retriever**
- Extend support to **multimodal inputs** (audio, video-based document querying)

---

## 📈 Outcomes

- ✅ 95%+ transcription accuracy
- ✅ Real-time performance
- ✅ High-quality, natural voice responses
- ✅ Strong accessibility for non-technical and hands-free users

---

## 📂 Repository Structure



├── app/                    # Streamlit app interface
├── embeddings/             # SentenceTransformer scripts
├── retriever/              # FAISS / DeepLake indexers
├── speech/                 # STT and TTS handlers
├── llm/                    # GPT-3.5 query interface
├── data/                   # Sample documents
└── README.md               # Project overview



---


