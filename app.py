import streamlit as st
import os
import time
from datetime import datetime
from pathlib import Path
from chatbot import RAGChatbot

# Page configuration
st.set_page_config(
    page_title="Live RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-ready {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        color: #065f46;
    }
    .status-updating {
        background-color: #dbeafe;
        border: 1px solid #3b82f6;
        color: #1e40af;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e0e7ff;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f3f4f6;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RAGChatbot(
        faq_file="faq.txt",
        model_type="ollama",  # Change to "openai" for GPT models
        model_name="llama2"
    )
    st.session_state.messages = []
    st.session_state.last_update = datetime.now()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<div class="main-header">🤖 Live RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280;">Real-Time FAQ & Knowledge Retrieval System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # FAQ file status
    st.subheader("📄 Knowledge Base")
    faq_path = Path("faq.txt")
    
    if faq_path.exists():
        file_stats = faq_path.stat()
        st.success("FAQ file found")
        st.caption(f"Size: {file_stats.st_size} bytes")
        st.caption(f"Modified: {datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("FAQ file not found")
        st.info("Creating default faq.txt...")
        default_faq = """Q: What is RAG?
A: RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them as context for generating accurate answers.

Q: How does the live update feature work?
A: The system monitors the FAQ file for changes. When detected, it automatically re-processes the content, creates new embeddings, and updates the vector store without requiring an application restart.

Q: What is FAISS?
A: FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It's used to quickly find the most relevant FAQ entries for a given query.

Q: Which LLM models are supported?
A: The system supports both Ollama (local models like Llama 2, Mistral) and OpenAI GPT models (GPT-3.5, GPT-4). You can configure which model to use in the settings.

Q: What are embeddings?
A: Embeddings are numerical vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling efficient similarity search.

Q: How accurate are the answers?
A: Answer accuracy depends on the quality of the FAQ content and the LLM model used. The RAG approach ensures answers are grounded in your knowledge base, reducing hallucinations.

Q: Can I use multiple documents?
A: Yes, the system can be extended to support multiple file formats including TXT, PDF, CSV, and DOCX files.

Q: What is the response time?
A: Typical response time is 1-3 seconds, depending on the LLM model, query complexity, and vector store size."""
        
        with open("faq.txt", "w") as f:
            f.write(default_faq)
        st.rerun()
    
    st.divider()
    
    # Vector store status
    st.subheader("🗄️ Vector Store")
    vector_status = st.session_state.chatbot.get_vector_store_status()
    
    if vector_status["initialized"]:
        st.markdown(f'<div class="status-box status-ready">✅ Ready<br/>Chunks: {vector_status["num_chunks"]}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-updating">⏳ Initializing...</div>', 
                   unsafe_allow_html=True)
    
    st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Manual refresh button
    if st.button("🔄 Refresh Vector Store", use_container_width=True):
        with st.spinner("Updating vector store..."):
            st.session_state.chatbot.load_and_index_documents()
            st.session_state.last_update = datetime.now()
            st.success("Vector store updated!")
            time.sleep(1)
            st.rerun()
    
    st.divider()
    
    # Model settings
    st.subheader("🔧 Model Settings")
    model_type = st.selectbox(
        "Model Type",
        ["ollama", "openai"],
        index=0 if st.session_state.chatbot.model_type == "ollama" else 1
    )
    
    if model_type == "ollama":
        model_name = st.selectbox(
            "Model Name",
            ["llama2", "mistral", "codellama", "phi"],
            index=0
        )
    else:
        model_name = st.selectbox(
            "Model Name",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0
        )
    
    if st.button("Apply Settings", use_container_width=True):
        st.session_state.chatbot.model_type = model_type
        st.session_state.chatbot.model_name = model_name
        st.session_state.chatbot.initialize_llm()
        st.success("Settings applied!")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Session Statistics")
    st.metric("Total Messages", len(st.session_state.messages))
    st.metric("Queries Processed", len([m for m in st.session_state.messages if m["role"] == "user"]))
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.header("💬 Chat Interface")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br/>{message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong><br/>{message["content"]}</div>', 
                       unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.text(f"Source {i}:")
                        st.code(source, language=None)

# Chat input
col1, col2 = st.columns([5, 1])

with col1:
    user_query = st.text_input(
        "Ask a question:",
        placeholder="e.g., What is RAG?",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    submit_button = st.button("Send 📤", use_container_width=True)

# Process query
if submit_button and user_query:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now()
    })
    
    # Get response
    with st.spinner("🤔 Thinking..."):
        try:
            response = st.session_state.chatbot.query(user_query)
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("source_documents", []),
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}",
                "timestamp": datetime.now()
            })
    
    st.rerun()

# Sample queries section
st.divider()
st.subheader("💡 Try These Sample Queries")

sample_queries = [
    "What is RAG?",
    "How does the live update feature work?",
    "Which LLM models are supported?",
    "What are embeddings?",
    "How accurate are the answers?",
    "What is FAISS?"
]

cols = st.columns(3)
for i, query in enumerate(sample_queries):
    with cols[i % 3]:
        if st.button(query, key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now()
            })
            
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.chatbot.query(query)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("source_documents", []),
                        "timestamp": datetime.now()
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>Live RAG Assistant v1.0 | Built with LangChain, FAISS & Streamlit</p>
    <p>💡 Edit <code>faq.txt</code> to update the knowledge base automatically</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh to detect FAQ changes (every 5 seconds)
if st.session_state.chatbot.check_for_updates():
    st.session_state.last_update = datetime.now()
    st.info("📄 FAQ file updated! Vector store refreshed automatically.")
    time.sleep(2)
    st.rerun()