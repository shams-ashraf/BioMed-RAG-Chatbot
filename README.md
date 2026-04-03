# 🚀 BioMed Advanced RAG Chatbot

An **enhanced Retrieval-Augmented Generation system** designed for accurate question answering over biomedical documents.  
Unlike traditional RAG pipelines, this system implements **iterative retrieval, self-correction, and context-aware response refinement** to improve answer quality in real-world scenarios.
---
🖼️ Demo
---

<img width="1280" height="721" alt="image" src="https://github.com/user-attachments/assets/c1d4a13e-facf-44e6-bb15-3b40dafab1a2" />

## 🎯 Overview
This project goes beyond basic RAG by introducing a **enhanced retrieval pipeline** that can:
- Evaluate its own answers  
- Dynamically refine search queries  
- Expand context progressively  
- Handle both structured and unstructured data  

The system ensures **high factual accuracy** by strictly grounding responses in document content.

---

## ✨ Key Features

### 🔍 Intelligent Retrieval
- Semantic search using embeddings (ChromaDB)  
- Separate pipelines for **text and table data**  
- Filtering low-quality chunks for cleaner context  

---

### 🔁 Iterative & Corrective RAG
- Multi-step retrieval instead of single-pass  
- Detects incomplete answers and **re-retrieves relevant data**  
- Expands search context dynamically around cited sources  

---

### 🧠 Self-Evaluation Mechanism
- The model evaluates whether the answer is:
  - Complete  
  - Partial  
  - Missing information  
- Triggers **corrective retrieval loops** when needed  

---

### 📊 Multimodal Support
- Handles:
  - Text  
  - Tables  
  - Structured document content  
- Prioritizes tables during retrieval for better factual grounding  

---

### 🧩 Context Engineering
- Context-aware chunking strategy  
- Progressive context expansion instead of full data injection  
- Token-aware trimming to stay within model limits  

---

### 📌 Source-Aware Reasoning
- Tracks which chunks were actually used in answers  
- Avoids redundant retrieval  
- Improves transparency and factual consistency  

---

### 💬 Conversational Memory
- Maintains short-term chat context  
- Uses **compressed chat history** for efficiency  

---

### ⚙️ Production-Oriented Design
- API key rotation to handle rate limits  
- Robust error handling (429, 413, retries)  
- Scalable modular architecture  

---

## 🧠 System Architecture

User Query  
→ Semantic Search (Text + Tables)  
→ Iterative Retrieval Loop  
→ Context Construction  
→ LLM Evaluation  
→ Self-Correction (if needed)  
→ Final Answer  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- ChromaDB  
- LLM API (Groq / OpenAI-compatible)  
- Embeddings (SentenceTransformers)  

---

## 📂 Project Structure
├── app.py # Streamlit UI
├── API.py # Main RAG loop + LLM calls
├── chat_engine/
│ ├── retrieval.py # Chunk retrieval
│ ├── iteration.py # Iterative logic
│ ├── utils.py # Helper functions
├── chroma_db/ # Vector database
├── styles.py # UI styling
├── requirements.txt


---

## 🚀 How It Works

1. Retrieve relevant chunks (text + tables)  
2. Build structured context  
3. Query the LLM  
4. Evaluate answer completeness  
5. If incomplete → refine retrieval  
6. Repeat until answer is complete  

---
## 💡 Key Learnings

- Designing **self-correcting AI systems**  
- Managing context within strict token limits  
- Building **robust RAG pipelines beyond simple retrieval**  
- Handling real-world edge cases in LLM applications  
