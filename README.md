# AskMyPDFs

Chat with multiple PDF documents using a conversational AI. Upload your PDFs, and ask questions in plain English. The app retrieves relevant content and answers using an LLM.

## How it works

1. PDFs are parsed and split into overlapping text chunks
2. Chunks are embedded using `hkunlp/instructor-xl` (runs locally via HuggingFace)
3. Embeddings are stored in a FAISS vector store for fast similarity search
4. Questions are answered by `llama-3.1-8b-instant` via Groq, with full conversation memory

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/AskMyPDFs.git
cd AskMyPDFs
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

## Usage

1. Upload one or more PDF files using the sidebar
2. Click **Upload** to process them (the button is disabled until files are selected)
3. Type a question in the chat input and press Enter

## Tech stack

| Component | Library |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Text splitting | LangChain `CharacterTextSplitter` |
| Embeddings | `hkunlp/instructor-xl` via HuggingFace (local) |
| Vector store | FAISS |
| LLM | Llama 3.1 8B via Groq (free) |
| Memory | LangChain `ConversationBufferMemory` |

## Requirements

- Python 3.9+
- A free [Groq API key](https://console.groq.com)
