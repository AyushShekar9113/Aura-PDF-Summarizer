# 🚀 Aura RAG — AI-Powered PDF Summarizer & Q&A

Aura RAG is an **AI-powered platform** that transforms how we interact with documents.  
It allows users to upload any PDF, get instant summaries, ask AI questions directly from the document, and even translate content — all in one place.

---

## 🌟 Features

✅ **AI Summarization** — Instantly generates concise, meaningful summaries from long PDFs.  
💬 **Interactive Q&A** — Ask questions directly from the uploaded document using an intelligent RAG pipeline.  
🌍 **Multilingual Translation** — Translate summaries or answers into multiple languages.  
📥 **Download Options** — Save summaries and chat history for later use.  
⚡ **Real-Time Processing** — Get results within seconds through a clean Streamlit interface.  
☁️ **Cloud Hosted (Streamlit)** — Always live and accessible anytime, anywhere.

---

## 🧠 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **LLM Engine** | Google Gemini (via LangChain) |
| **Framework** | Streamlit |
| **Embeddings** | HuggingFace Transformers |
| **Vector Store** | FAISS |
| **Backend Logic** | LangChain RAG Pipeline |
| **Deployment** | Streamlit Cloud |
| **Languages** | Python |

---

## 🛠️ How It Works

1. Upload a PDF 📄  
2. The system extracts text and splits it into chunks  
3. Each chunk is embedded using **HuggingFace embeddings**  
4. The embeddings are stored in **FAISS** for semantic search  
5. When you ask a question, **LangChain + Gemini** retrieves and answers contextually  
6. Streamlit handles UI for real-time interaction and multilingual translation  

---

## ⚙️ Installation & Setup

You can run Aura RAG locally by following these steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AyushShekar9113/Aura-PDF-Summarizer.git
cd Aura-PDF-Summarizer
