# 🤗 Comfort Zone - Mental Health Chatbot

A compassionate AI-powered mental health assistant that uses **RAG (Retrieval-Augmented Generation)** to provide informed, supportive conversations. This project combines a FastAPI backend with a Streamlit frontend to create a safe, empathetic space for mental health discussions.

## 🧠 RAG Architecture

This project implements a sophisticated **RAG pipeline** that enhances the AI's responses with relevant, verified mental health information:

### RAG Components:

1. **📚 Knowledge Base Construction**
   - Curated mental health resources from trusted websites and PDFs
   - Web scraping and PDF text extraction for comprehensive coverage
   - Structured data from `mental_health_resources.json`

2. **🔍 Vector Embedding & Retrieval**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` for text embeddings
   - ChromaDB vector store for efficient similarity search
   - Context-aware document retrieval based on user queries

3. **💬 Generation with Context**
   - Groq's LLaMA 3 70B model for response generation
   - Conversational memory maintained across sessions
   - Empathetic prompt engineering for mental health support

## 🌟 Features

- **🧠 RAG-Powered Responses**: Combines LLM capabilities with verified mental health knowledge
- **Bilingual Support**: Full Arabic/English translation capabilities
- **Context-Aware Conversations**: Remembers chat history and provides relevant responses
- **Rich Knowledge Base**: Scrapes and processes mental health resources from trusted sources
- **Secure & Private**: Local chat storage with session management
- **Professional Boundaries**: Includes safety disclaimers and crisis guidance

## 🏗️ Project Structure

```
mental-health-chatbot/
├── healthy.py           # FastAPI backend with RAG pipeline
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

1. **Start the Backend Server** (RAG Pipeline)
   ```bash
   uvicorn healthy:app --reload --port 8000
   ```
   The API will be available at `http://127.0.0.1:8000`

2. **Start the Frontend Interface**
   ```bash
   streamlit run frontend.py
   ```
   The web interface will open at `http://localhost:8501`

## 🔧 RAG Configuration

### Backend RAG Settings (healthy.py)
- **GROQ_API_KEY**: Your Groq API key for LLM generation
- **LLM_MODEL**: "llama3-70b-8192" for response generation
- **EMBEDDING_MODEL**: "sentence-transformers/all-MiniLM-L6-v2" for vector embeddings
- **VECTOR_DB_PATH**: "./chroma_db" - persistent vector storage
- **Retrieval Method**: Similarity search with conversation context

### RAG Data Processing
```python
# Document processing pipeline
1. Web scraping & PDF extraction → 2. Text chunking → 
3. Vector embedding → 4. ChromaDB storage → 
5. Context retrieval → 6. Augmented generation
```

## 📚 Knowledge Base & Data Sources

The RAG system uses curated mental health resources including:
- Positive psychology practices
- CBT self-help workbooks
- Mindfulness exercises
- Youth mental health guides
- Professional mental health FAQs
- Trusted PDF resources from mental health organizations

## 🔍 RAG Pipeline Details

### 1. **Knowledge Ingestion**
```python
# Extract and process multiple data sources
- JSON resource files → Structured mental health information
- Web scraping → Latest articles and FAQs  
- PDF processing → Professional guides and workbooks
```

### 2. **Vectorization & Storage**
```python
# Create embeddings and store in vector database
embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma.from_documents(documents, embeddings)
```

### 3. **Retrieval & Generation**
```python
# RAG chain implementation
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=vector_db.as_retriever(),
    memory=conversation_memory,
    combine_docs_chain_kwargs={"prompt": empathetic_prompt}
)
```

## 🛡️ Safety Features in RAG

- **Verified Sources**: All retrieved information comes from trusted mental health resources
- **Crisis Detection**: RAG-enhanced responses include professional guidance
- **Contextual Safety**: Prompt engineering ensures appropriate boundaries
- **No Hallucinations**: Grounded responses based on actual mental health content

## 🌐 Multilingual RAG Support

- **Arabic/English Detection**: Automatic language identification
- **Translation Pipeline**: 
  - Arabic input → English (for RAG processing) → Arabic output
  - Maintains context and empathy across languages
- **Cultural Sensitivity**: RAG responses consider linguistic nuances

## 💾 RAG Data Persistence

- **Vector Database**: Persistent ChromaDB stores all embedded knowledge
- **Chat Sessions**: Conversation history for contextual retrieval
- **Resource Cache**: Local storage of scraped content for fast access

## 🎯 RAG Benefits in Mental Health Context

1. **Accuracy**: Responses grounded in verified mental health information
2. **Relevance**: Context-aware retrieval based on conversation history  
3. **Consistency**: Maintains professional boundaries and safety guidelines
4. **Comprehensiveness**: Draws from diverse mental health resources
5. **Personalization**: Adapts responses based on user's conversation context

## 🔌 API Documentation

Once running, visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### RAG-Enhanced Chat Endpoint
- `POST /chatbot`: Main conversation endpoint with RAG
  ```json
  {
    "question": "user message",
    "chat_history": [["user_msg1", "ai_msg1"], ...]
  }
  ```

## ⚠️ Important Notes

- This is an educational tool using RAG for informed responses, not a substitute for professional mental healthcare
- The RAG system ensures responses are based on verified mental health information
- For urgent concerns, please contact licensed healthcare providers
- All conversations use RAG to provide contextually appropriate support

## 🔍 Technical RAG Implementation

### Retrieval Process:
1. **Query Understanding**: Analyze user question and conversation context
2. **Vector Search**: Find most relevant mental health documents
3. **Context Augmentation**: Combine retrieved documents with conversation history
4. **Informed Generation**: Generate response using both LLM knowledge and retrieved content

### Key RAG Features:
- **Conversational Memory**: Maintains context across multiple exchanges
- **Multi-source Retrieval**: Combines structured and unstructured data
- **Real-time Processing**: Dynamic retrieval based on current conversation
- **Quality Assurance**: Verified sources reduce misinformation risk

---

*Remember: Your mental health matters. This RAG-powered tool provides informed support, but professional help is important for serious concerns.*
