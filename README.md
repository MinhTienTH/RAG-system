# PDF Chatbot 📄💬

## Overview

A Streamlit-powered web application that allows users to upload PDF documents and interactively ask questions about their content using advanced natural language processing.

## Features

- 📤 PDF Document Upload
- 🤖 AI-Powered Q&A
- 💾 Persistent Chat History
- 🔍 Contextual Document Retrieval

## Prerequisites

- Python 3.8+
- Hugging Face Account (for API token)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MinhTienTH/RAG-system.git
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Hugging Face API Token:
- Create a `.env` file in the project root
- Add your Hugging Face token: 
```
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token_here
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### How to Use
1. Upload a PDF document
2. Wait for processing confirmation
3. Ask questions about the document in the chat interface

## Technologies Used

- **Framework**: Streamlit
- **Language Model**: Google FLAN-T5 Base
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Store**: Chroma
- **NLP Libraries**: LangChain, HuggingFace Transformers

## Configuration

Customize model parameters in `get_conversation_chain()`:
- Adjust `temperature`
- Modify `max_length`
- Change retrieval settings
# RAG-system
