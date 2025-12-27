# Document-Based Question Answering System

This project is a Document-Based Question Answering (QA) system that allows users to upload a PDF or text document and ask questions strictly based on the document’s content.

The system uses semantic embeddings and FAISS vector similarity search to retrieve relevant document context and provide grounded answers. It explicitly avoids hallucination by rejecting questions that are outside the scope of the uploaded document.

## Features

- Upload PDF or TXT documents
- Automatic document chunking and indexing
- Semantic search using dense vector embeddings
- Fast similarity search with FAISS
- Context-aware question answering
- Explicit out-of-scope detection
- Simple Flask-based web interface

## Tech Stack

- Python
- Flask
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS
- LangChain
- HTML / Jinja2

## Project Structure

```
miko_rag/
├── src/
│   ├── app.py               # Flask application
│   ├── retriever.py         # FAISS retrieval logic
│   └── ingest_runtime.py    # Document ingestion and indexing
├── data/
│   └── uploads/             # Uploaded documents
├── templates/
│   └── index.html           # Web UI
├── requirements.txt
└── README.md
```


## Installation

1. Clone the repository

git clone https://github.com/<your-username>/document-qa-faiss.git
cd document-qa-faiss

2. Create and activate virtual environment

python -m venv venv

Windows:
venv\Scripts\activate

Linux / macOS:
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

## Run the Application

cd src
python app.py

Open the browser and visit:
http://127.0.0.1:5000

## How It Works

1. User uploads a document (PDF or TXT)
2. The document is split into smaller text chunks
3. Each chunk is converted into a vector embedding
4. Embeddings are indexed using FAISS
5. User asks a question
6. The question is embedded and matched against document embeddings
7. Relevant chunks are retrieved
8. The top chunk is shown as the Final Answer
9. Additional chunks are shown as Supporting Context

## Hallucination Prevention

If a question cannot be answered from the uploaded document, the system responds:

"The uploaded document does not contain information to answer this question."

This ensures all answers remain grounded and reliable.

## Example Questions

- What is this document about?
- What text formatting examples are mentioned?
- What items are in the unordered list?
- What does the table contain?

## Future Improvements

- LLM-based answer summarization
- Confidence scoring for answers
- Multi-document indexing
- Improved UI styling

## License

This project is intended for educational and demonstration purposes.

