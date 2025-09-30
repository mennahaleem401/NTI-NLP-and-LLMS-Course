# 🤗 Comfort Zone - Mental Health Chatbot

A compassionate AI-powered mental health assistant that provides supportive conversations and resources. This project combines a FastAPI backend with a Streamlit frontend to create a safe, empathetic space for mental health discussions.

## 🌟 Features

- **Bilingual Support**: Full Arabic/English translation capabilities
- **Context-Aware Conversations**: Remembers chat history and provides relevant responses
- **Rich Knowledge Base**: Scrapes and processes mental health resources from trusted sources
- **Secure & Private**: Local chat storage with session management
- **Professional Boundaries**: Includes safety disclaimers and crisis guidance

## 🏗️ Project Structure

```
mental-health-chatbot/
├── healthy.py           # FastAPI backend server
├── frontend.py          # Streamlit web interface
├── mental_health_resources.json  # Curated mental health resources
├── chroma_db/           # Vector database (auto-generated)
├── pdfs/                # Downloaded PDF resources (auto-generated)
└── chats.json          # User chat history (auto-generated)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API account (for LLM access)
- LangSmith account (optional, for tracing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mental-health-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   export LANGCHAIN_API_KEY="your_langsmith_api_key_here"  # Optional
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   uvicorn healthy:app --reload --port 8000
   ```
   The API will be available at `http://127.0.0.1:8000`

2. **Start the Frontend Interface**
   ```bash
   streamlit run frontend.py
   ```
   The web interface will open at `http://localhost:8501`

## 🔧 Configuration

### Backend Settings (healthy.py)
- **GROQ_API_KEY**: Your Groq API key for LLM access
- **LLM_MODEL**: Default "llama3-70b-8192"
- **EMBEDDING_MODEL**: "sentence-transformers/all-MiniLM-L6-v2"
- **VECTOR_DB_PATH**: "./chroma_db" - vector storage location

### Frontend Settings (frontend.py)
- **API_URL**: Backend endpoint (default: "http://127.0.0.1:8000/chatbot")
- **CHAT_FILE**: "chats.json" - local chat storage

## 📚 Data Sources

The chatbot uses curated mental health resources including:
- Positive psychology practices
- CBT self-help workbooks
- Mindfulness exercises
- Youth mental health guides
- Professional mental health FAQs

## 🛡️ Safety Features

- **Crisis Detection**: Gently guides users to professional help when needed
- **Professional Boundaries**: Clear disclaimers about not being a licensed therapist
- **No Diagnosis**: Explicitly avoids medical diagnosis or treatment recommendations
- **Empathetic Responses**: Trained to provide supportive, non-judgmental conversations

## 🌐 Language Support

- **Automatic Detection**: Identifies Arabic/English input
- **Seamless Translation**: Processes all content in English, translates responses back
- **Bilingual UI**: Supports both languages in conversation flow

## 💾 Data Persistence

- **Chat Sessions**: Saved locally in JSON format
- **Vector Database**: Persistent ChromaDB for fast retrieval
- **Resource Cache**: Downloaded PDFs and web content

## 🔌 API Documentation

Once running, visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Key Endpoints

- `POST /chatbot`: Main conversation endpoint
  ```json
  {
    "question": "user message",
    "chat_history": [["user_msg1", "ai_msg1"], ...]
  }
  ```

## 🎯 Usage

1. **Create a New Session**: Use the sidebar to start a new chat
2. **Conversation**: Type your message and receive empathetic responses
3. **Session Management**: Switch between different conversation threads
4. **History**: All chats are saved and can be revisited

## ⚠️ Important Notes

- This is an educational tool, not a substitute for professional mental healthcare
- For urgent concerns, please contact licensed healthcare providers
- All conversations are stored locally for continuity
- The system includes appropriate crisis response guidance

## 🔍 Technical Details

### Architecture
- **Backend**: FastAPI with LangChain for RAG pipeline
- **Frontend**: Streamlit for interactive web interface
- **Vector Store**: ChromaDB with HuggingFace embeddings
- **LLM**: Groq API with LLaMA 3 70B model
- **Translation**: Helsinki-NLP models for Arabic/English

### Processing Pipeline
1. Web scraping and PDF extraction from trusted sources
2. Text chunking and vector embedding
3. Context-aware retrieval augmented generation
4. Bilingual translation layer
5. Empathetic response generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is intended for educational and supportive purposes. Please ensure compliance with all API terms of service and mental health guidelines.

