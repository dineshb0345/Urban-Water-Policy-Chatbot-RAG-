Urban Water Policy RAG Chatbot:

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to query and understand **urban water policy documents** through natural language.  
The system retrieves relevant sections from official policy PDFs and generates **accurate, grounded responses** using a **locally hosted Large Language Model (LLaMA 3 via Ollama)**.



Problem Statement:

Urban water policies are often lengthy, technical, and difficult for citizens, students, and officials to interpret quickly.  
There is a need for an **interactive, reliable, and document-grounded assistant** that can answer policy-related questions without hallucination.



Objectives:

Enable natural language querying of water policy documents  
Ensure answers are **strictly grounded in official PDFs**  
Avoid hallucinations and opinions  
Provide **clean citations** (page-level references)  
Work **offline without paid APIs**  
Offer a conversational chatbot experience  



Solution Overview:

This project implements a Retrieval-Augmented Generation (RAG) pipeline:

1. Policy PDFs are loaded and split into chunks  
2. Text chunks are converted into vector embeddings  
3. FAISS is used for fast semantic search  
4. Relevant document chunks are retrieved per query  
5. A local LLaMA 3 model generates responses **only from retrieved content**  
6. The chatbot presents answers with **source citations**  


System Architecture:

User (Browser)
     ↓
Streamlit Chat UI
     ↓
FAISS Vector Store (Policy Chunks)
     ↓
Retriever (Semantic Search)
     ↓
LLaMA 3 (Ollama - Local LLM)
     ↓
Grounded Answer + Sources


Tech Stack:

**Frontend / UI**: Streamlit  
**LLM**: LLaMA 3 (via Ollama – local inference)  
**Framework**: LangChain (modern Runnable API)  
**Vector Database**: FAISS  
**Embeddings**: Sentence Transformers  
**Language**: Python 


Key Features:

Upload and process multiple policy PDFs  
Chat-style conversational interface  
Context-aware answers (RAG)  
Page-level source citations  
No hallucinated responses  
Offline & privacy-friendly (no cloud APIs)  
Fast semantic search using FAISS


Types of Questions Supported:

Policy objectives and scope  
Rules and regulations  
Technical specifications (IS / ISO standards)  
Penalties and enforcement clauses  
Procedures and application processes  
Document summaries (e.g., *“Summarize in 10 lines”*)  
