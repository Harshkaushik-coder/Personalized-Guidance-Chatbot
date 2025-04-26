# Import required libraries
import streamlit as st
import time
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Streamlit page configuration
st.set_page_config(page_title="Career Suggestion Chatbot", page_icon="💼", layout="centered")

# Load vectorstore with embeddings
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="career_db", embedding_function=embedding_model)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_vectorstore()

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "bot", "text": "👋 Hi! Tell me your interests or skills and I’ll suggest career options!"}
    ]

# App title and subtitle
st.title("🎓 Career Personalised Chatbot")
st.caption("💡 Discover career options based on your interests.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='stChatMessage {msg['role']}'>{msg['text']}</div>", unsafe_allow_html=True)

# Chat input from user
user_input = st.chat_input("Type your interest or skill...")

if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(f"<div class='stChatMessage user'>{user_input}</div>", unsafe_allow_html=True)

    with st.spinner("🔍 Thinking..."):
        time.sleep(0.7)
        results = retriever.get_relevant_documents(user_input)
        suggestions = [doc.page_content for doc in results]

    # Generate bot response
    if suggestions:
        reply = "🔎 *Based on your input, here are some career suggestions:*"
        formatted = "\n".join([f"<div class='career-card'>✅ {s}</div>" for s in suggestions])
        reply += f"\n\n{formatted}"
    else:
        reply = "❌ Sorry, I couldn't find any relevant career suggestions."

    st.session_state.messages.append({"role": "bot", "text": reply})
    with st.chat_message("bot"):
        st.markdown(f"<div class='stChatMessage bot'>{reply}</div>", unsafe_allow_html=True)
