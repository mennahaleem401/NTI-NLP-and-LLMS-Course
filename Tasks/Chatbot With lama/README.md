# Chatbot with FastAPI Backend and Streamlit Frontend

A simple chatbot application using Groq's LLaMA model, featuring a FastAPI backend and Streamlit frontend interface.

## Project Structure

```
chatbotTask/
├── fastapibackend.py    # FastAPI backend server
├── streamlitapp.py      # Streamlit frontend application
└── .env                 # Environment variables (not committed)
```

## Prerequisites

- Python 3.7+
- Groq API account ([sign up here](https://groq.com/))
- pip (Python package manager)

## Installation

1. Clone or create the project directory
2. Install required dependencies:
```bash
pip install fastapi uvicorn groq streamlit requests python-dotenv
```

3. Create a `.env` file in the project root with your Groq API key:
```env
GROQ_API_KEY="your_actual_api_key_here"
```

## Usage

1. Start the FastAPI backend server:
```bash
uvicorn fastapibackend:app --reload
```
The backend will be available at `http://127.0.0.1:8000`

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run streamlitapp.py
```
The chat interface will open in your browser at `http://localhost:8501`

3. Type messages in the chat input to interact with the LLaMA-3.3-70b model

## API Endpoint

The backend exposes a POST endpoint at `/chat` that accepts JSON in the format:
```json
{"message": "Your message here"}
```

And returns responses in the format:
```json
{"reply": "Bot's response here"}
```

## Features

- Clean Streamlit chat interface
- FastAPI backend with Groq integration
- Chat history maintained during session
- Error handling for API failures
- Responsive design with loading states

## Notes

- Remember to keep your API key confidential
- The `.env` file is included in `.gitignore` by default
- Modify the model parameters in `fastapibackend.py` to use different Groq models
- Adjust `max_tokens` in the backend to control response length
```

To use this chatbot:
1. Replace `"your_actual_api_key_here"` with your actual Groq API key
2. Ensure both servers are running simultaneously
3. The chat interface will automatically connect to the backend API


