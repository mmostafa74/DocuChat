# 📄 DocuChat: RAG-based Document QA with Streamlit & OpenRouter

**DocuChat** is a lightweight Streamlit web app that lets you upload documents (PDF, TXT, CSV, DOCX), process them with LangChain, and ask questions using OpenRouter-hosted LLMs via a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features

* 🧐 **RAG pipeline** using FAISS vector store & HuggingFace embeddings
* 📄 **Multi-format document support**: PDF, TXT, CSV, DOC/DOCX
* 🔍 **Semantic search** and context-aware answers
* 🌐 **OpenRouter API integration** (ChatGPT, Claude, etc.)
* 🎛️ **Streamlit UI** with configurable sidebar and styling
* ✅ Plug-and-play setup with `.env` and `config.toml`

---

## 📁 Project Structure

```text
.
├── app.py                    # Streamlit app entrypoint
├── config.py                 # Configuration loader with .env + TOML support
├── utils.py                  # Core logic: loading, embedding, retrieval
├── config.toml               # UI, app, and chat settings
├── .env                      # Environment variables (API keys, etc.)
├── chroma_db/                # Chroma/FAISS vector store (created at runtime)
├── docs/                     # Optional source docs directory
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/mmostafa74/DocuChat.git
cd DocuChat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key
```

### 4. (Optional) Edit `config.toml`

```toml
[app]
title = "DocuChat"
description = "Upload documents and get answers using AI"
page_icon = "🤖"

[ui]
layout = "wide"
initial_sidebar_state = "expanded"

[sidebar]
chat_controls_title = "## ⚙️ Chat Controls"
chat_stats_title = "## 📊 Chat Statistics"
clear_button_text = "🗑️ Clear Chat"
export_button_text = "📥 Export Chat"
input_placeholder = "Type your message here..."
```

---

## 🧠 How It Works

1. **Upload Files** → Supports `.pdf`, `.txt`, `.csv`, `.docx`
2. **Chunking** → Documents are split into overlapping sections
3. **Embedding** → Uses `sentence-transformers/all-MiniLM-L6-v2`
4. **Indexing** → Stores vectors in FAISS
5. **Querying** → Queries are matched to top-K chunks
6. **Prompting** → Retrieved context + user query → LLM prompt
7. **Answering** → OpenRouter LLM generates the final response

---

## 📤 Embedding + Retrieval Logic

From `utils.py`:

* `load_documents_from_files(...)`: loads and cleans files
* `create_vector_store(...)`: chunks and embeds using FAISS
* `get_relevant_context(...)`: retrieves top-k similar chunks
* `create_rag_prompt(...)`: injects context into a system prompt

---

## 💡 Example Prompt

```text
Based on the following context from the uploaded documents, please answer the question.

Context:
[...top-k content...]

Question: What was the net profit in 2023?

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please mention that.
```

---

## 🔐 API Support

Supports any OpenRouter-compatible model via:

* `OPENROUTER_API_KEY` in `.env`
* Models can include OpenAI (gpt-3.5/gpt-4), Claude, Mistral, etc.

---

## 🧪 Running Locally

```bash
streamlit run app.py
```

---

## 📦 Dependencies

* `streamlit`
* `langchain`
* `faiss-cpu`
* `sentence-transformers`
* `python-dotenv`
* `toml`

Add others as needed in `requirements.txt`.

---

## 🛡️ License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [OpenRouter](https://openrouter.ai/)
* [Streamlit](https://streamlit.io/)
* [Sentence Transformers](https://www.sbert.net/)

---

Built with ❤️ using Streamlit and OpenRouter
