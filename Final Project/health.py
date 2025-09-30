import os
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from transformers import pipeline

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "Write Yor API")
LANGCHAIN_API_KEY = os.environ.get("Write Yor API")
LANGCHAIN_PROJECT = "Mental Health Chatbot"
JSON_FILE_PATH = "mental_health_resources.json"
JSON_FILE_PATH2="F_Q.json"
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-70b-8192"

# --- LangSmith Settings ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Mental Health Chatbot API",
    description="An API for a compassionate mental health assistant."
)

# --- Pydantic Models for API ---
class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str

# Initialize the translation models when the application starts
translator_ar_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en")
translator_en_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")

def detect_language(text: str) -> str:
    """Detect if input is Arabic using a simple character range check."""
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "ar"
    return "en"

def translate_to_en(text: str) -> str:
    """Translate Arabic text to English."""
    return translator_ar_en(text)[0]['translation_text']

def translate_to_ar(text: str) -> str:
    """Translate English text to Arabic."""
    return translator_en_ar(text)[0]['translation_text']


# --- Web Scraping and PDF Extraction Functions ---
def extract_text_from_pdf(url, save_dir="pdfs"):
    try:
        Path(save_dir).mkdir(exist_ok=True)
        local_path = Path(save_dir) / url.split("/")[-1]
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        with open(local_path, "wb") as f: f.write(response.content)
        text = ""
        with pdfplumber.open(local_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
        print(f"📄 PDF loaded: {url} ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"⚠️ Could not process PDF {url}: {e}")
        return None

from urllib.parse import urljoin

def clean_html(url):
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

       
        pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith(".pdf")]
        if pdf_links:
            all_pdf_text = ""
            for pdf_url in pdf_links:
                if pdf_url.startswith("/"):  
                    pdf_url = urljoin(url, pdf_url)
                pdf_text = extract_text_from_pdf(pdf_url)
                if pdf_text:
                    all_pdf_text += pdf_text + "\n"
            if all_pdf_text:
                print(f"📄 Extracted {len(pdf_links)} PDFs from {url}")
                return all_pdf_text

      
        faqs = []
       
        for strong_tag in soup.find_all("strong"):
            question = strong_tag.get_text(" ", strip=True)
            
            if question.endswith("?") or question.lower().startswith("q"):
                answer_tag = strong_tag.find_next("p")
                if answer_tag:
                    answer = answer_tag.get_text(" ", strip=True)
                    faqs.append(f"Q: {question}\nA: {answer}")

        
        for header in soup.find_all(["h2", "h3"]):
            question = header.get_text(" ", strip=True)
            if question.endswith("?"):
                answer_tag = header.find_next("p")
                if answer_tag:
                    answer = answer_tag.get_text(" ", strip=True)
                    faqs.append(f"Q: {question}\nA: {answer}")

        if faqs:
            print(f"❓ Extracted {len(faqs)} Q&A pairs from {url}")
            return "\n\n".join(faqs)

        for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(paragraphs)
        print(f"🌐 Web loaded: {url} ({len(text)} chars)")
        return text.strip()

    except Exception as e:
        print(f"⚠️ Could not scrape {url}: {e}")
        return None



# --- Core Chatbot Logic ---

@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTOR_DB_PATH):
        print("Vector database not found. Creating a new one...")
        create_vector_db()
    else:
        print("Existing vector database found.")
    app.state.embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)
    app.state.vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=app.state.embeddings)
    app.state.llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)
    print("Chatbot is initialized and ready.")

def create_vector_db():
    """Create vector DB by scraping web links and PDFs."""
    with open(JSON_FILE_PATH, "r", encoding='utf-8') as f: data= json.load(f)
    resource_docs, links_to_scrape = [], []
    for item in data:
        resource_docs.append(Document(
            page_content=f"Title: {item['Title']}\nDescription: {item['Description']}\nCategory: {item['Category']}",
            metadata={"source": item.get("Link", "JSON")}
        ))
        if item.get("Link"): links_to_scrape.append(item["Link"])
    print(f"🔗 Found {len(links_to_scrape)} links in resources to process...")
    scraped_docs = []
    for url in links_to_scrape:
        text_content = extract_text_from_pdf(url) if url.lower().endswith(".pdf") else clean_html(url)
        if text_content: scraped_docs.append(Document(page_content=text_content, metadata={"source": url}))
    all_documents = resource_docs + scraped_docs
    print(f"Total documents to be embedded: {len(all_documents)}")
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    vector_db.persist()
    print("🎉 ChromaDB created and data saved (JSON + Web/PDF content)")
    return vector_db

def get_qa_chain(llm, vector_db, chat_history):
    """Setup the conversational retrieval chain with a static, empathetic prompt."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for human_msg, ai_msg in chat_history:
        memory.chat_memory.add_user_message(human_msg)
        memory.chat_memory.add_ai_message(ai_msg)
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate and empathetic AI companion. Your primary goal is to be a supportive listener, making the user feel heard, validated, and understood. Listen first, be human-like, offer gentle suggestions only when appropriate, and ask open-ended questions.
---
**SAFETY CRITICAL:** You are NOT a licensed therapist. You MUST NEVER diagnose conditions. If the user mentions a crisis, you MUST gently guide them to seek help from a professional.
---
Context:
{context}
Chat History:
{chat_history}
User: {question}
Chatbot:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'chat_history', 'question'])
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}, verbose=True
    )

# --- API Endpoint with Translation Logic ---
@app.post("/chatbot", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a question, handles translation, and returns the chatbot's answer.
    """
    user_lang = detect_language(request.question)
    query = request.question
    
    # Translate Arabic input to English for the main LLM
    if user_lang == "ar":
        print("--> Detected language: Arabic. Translating to English...")
        query = translate_to_en(query)
    else:
        print("--> Detected language: English.")

    # Prepare and run the main conversational chain (always in English)
    qa_chain = get_qa_chain(app.state.llm, app.state.vector_db, request.chat_history)
    response = qa_chain({"question": query})
    answer = response['answer']

    # Translate the English answer back to Arabic if needed
    if user_lang == "ar":
        print("--> Translating response back to Arabic...")
        # Use text splitter to handle long responses safely
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = text_splitter.split_text(answer)
        translated_chunks = [translate_to_ar(chunk) for chunk in chunks]
        answer = " ".join(translated_chunks)

    return ChatResponse(answer=answer)