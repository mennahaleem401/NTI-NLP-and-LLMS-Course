import streamlit as st
import time
import requests

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"bot", "content": str}

st.title("💬 Simple Streamlit Chatbot")

# --- Display past messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# --- Get user input ---
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display thinking spinner
    with st.spinner("Thinking..."):
        time.sleep(0.5)  # simulate processing
        try:
            # Send request to FastAPI backend
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"message": user_input}
            )
            bot_reply = response.json()["reply"]
        except Exception as e:
            bot_reply = f"⚠️ Error: {e}"

    # Save bot reply
    st.session_state.messages.append({"role": "bot", "content": bot_reply})

    # Rerun so new messages appear immediately
    st.rerun()
