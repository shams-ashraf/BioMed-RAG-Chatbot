---
🚀 BioMed RAG Chatbot
---

An AI-powered document-based chatbot built using Retrieval-Augmented Generation (RAG). The system enables users to ask questions and receive accurate answers directly from official documents such as academic regulations and course materials.

---
🖼️ Demo
---

<img width="1280" height="721" alt="image" src="https://github.com/user-attachments/assets/c1d4a13e-facf-44e6-bb15-3b40dafab1a2" />

---
✨ Features
---
Answers are grounded strictly in document content (no hallucination)

Supports both unstructured text and structured table data

Iterative retrieval pipeline for improved answer accuracy

Context expansion using nearby document sections

Source tracking to avoid redundant retrieval

Efficient context management within token limits

Real-time interaction using Streamlit interface

---
🧠 Technical Highlights
---
Semantic search using embeddings (ChromaDB)

Separate pipelines for text and table processing

Iterative retrieval strategy:

    Retrieve relevant chunks
    
    Dynamically build context
    
    Query LLM
    
    Evaluate completeness
    
    Refine results if needed
    
Progressive context expansion instead of full data injection

Lightweight chat history compression

API key rotation for handling rate limits

Robust error handling:

    Rate limiting (429)
    
    Payload size limits (413)

----
⚙️ Tech Stack
----

Python

Streamlit

ChromaDB

OpenAI API (or LLM API)

NLP / Embeddings

---
📂 Project Structure
---
.
├── app.py

├── API.py

├── chat_engine/

├── chroma_db/

├── styles.py

├── requirements.txt

---
🚀 How to Run
---

pip install -r requirements.txt

streamlit run app.py

---
💡 What I Learned
---
This project helped me design intelligent AI systems that combine retrieval, context management, and iterative refinement to improve answer quality in real-world applications.
