import streamlit as st
import requests
import json
import os
from typing import List, Tuple

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/chatbot"
CHAT_FILE = "chats.json"   # Location to save chats

# --- Helper Functions ---
def load_sessions():
    """Load saved chats from a JSON file."""
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {} # Return empty dict if file is corrupted or empty
    return {}

def save_sessions(sessions):
    """Save all chat sessions to a JSON file."""
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=4)

# --- Page Setup ---
st.set_page_config(page_title="Comfort Zone", layout="wide")
st.title("🤗 Comfort Zone")
st.markdown(
    "Your safe space to talk, reflect, and find support. I'm here to listen without judgment."
)
# Add a disclaimer for safety
# st.warning(
#     "*Disclaimer:* This is an educational tool and not a substitute for a licensed professional. "
#     "For any urgent concerns, please consult with a qualified healthcare provider."
# )

# --- Session State Management ---
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
if "current_session" not in st.session_state:
    st.session_state.current_session = None

# --- Sidebar for Chat History ---
with st.sidebar:
    st.header("💬 Chat History")

    with st.form("new_session_form", clear_on_submit=True):
        session_name = st.text_input("Start a new chat:", key="new_session")
        submitted = st.form_submit_button("➕ Create New Chat")

    if submitted:
        name = (session_name or "").strip()
        if name:
            if name not in st.session_state.sessions:
                st.session_state.sessions[name] = [] # Start with an empty message list
                st.session_state.current_session = name
                save_sessions(st.session_state.sessions)
                st.rerun()
            else:
                st.warning("Session already exists. Please choose a different name.")

    if st.session_state.sessions:
        st.subheader("Previous Chats")
        for name in list(st.session_state.sessions.keys()): # Use list() to avoid issues when deleting
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(name, key=f"open_{name}", use_container_width=True):
                    st.session_state.current_session = name
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"delete_{name}", use_container_width=True):
                    del st.session_state.sessions[name]
                    if st.session_state.current_session == name:
                        st.session_state.current_session = None
                    save_sessions(st.session_state.sessions)
                    st.rerun()

# --- Main Chat Interface ---
if not st.session_state.current_session:
    st.info("👈 Create or select a session from the sidebar to start chatting.")
else:
    # Display previous messages from the selected session
    for msg in st.session_state.sessions[st.session_state.current_session]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("Type your message here and press Enter..."):
        current_chat = st.session_state.sessions[st.session_state.current_session]
        
        # Add user message to the chat history and display it
        current_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- THE FIX IS HERE: Correctly Prepare and Send the API Request ---
        with st.spinner("Thinking..."):
            bot_reply = ""
            try:
                # 1. Prepare the chat history for the API in the correct format
                chat_history_for_api: List[Tuple[str, str]] = []
                # Iterate through all messages except the newest one to form the history
                for i in range(0, len(current_chat) - 1, 2):
                    user_msg = current_chat[i]["content"]
                    ai_msg = current_chat[i+1]["content"]
                    chat_history_for_api.append((user_msg, ai_msg))

                # 2. Construct the full payload with the question AND the history
                payload = {
                    "question": prompt,
                    "chat_history": chat_history_for_api # <-- Sending the real history
                }

                # 3. Make the POST request
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # A better way to handle HTTP errors

                # 4. Extract the bot's reply using the correct key "answer"
                bot_reply = response.json()["answer"]

            except requests.exceptions.ConnectionError:
                bot_reply = f"❌ Connection Error: The backend at {API_URL} appears to be offline."
            except Exception as e:
                bot_reply = f"⚠ An unexpected error occurred: {e}"

        # Add the bot's response to the chat history
        current_chat.append({"role": "assistant", "content": bot_reply})
        
        # Save the entire updated session history to the file
        save_sessions(st.session_state.sessions)
        
        # Rerun the script to display the new message immediately
        st.rerun()