# 🤖 Live RAG Assistant

**Real-Time FAQ and Knowledge Retrieval System**

A production-ready Retrieval-Augmented Generation (RAG) system that provides accurate answers to user queries with **automatic knowledge base updates** without requiring app restarts.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.6-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## ✨ Key Features

### 🔄 Live Knowledge Updates
- **Zero Downtime**: Modify `faq.txt` while the app runs
- **Automatic Detection**: System detects file changes instantly
- **Smart Refresh**: Updates vector embeddings without restart
- **Real-Time Sync**: Changes reflected in next query

### 🧠 Advanced RAG Pipeline
- **Semantic Search**: FAISS vector store for fast retrieval
- **Context-Aware**: Retrieves most relevant knowledge chunks
- **Source Citations**: Shows which FAQ entries were used
- **Hybrid Models**: Support for Ollama (local) and OpenAI (cloud)

### 🎨 Modern UI
- **Chat Interface**: Intuitive Streamlit-based design
- **Live Status**: Real-time vector store monitoring
- **Sample Queries**: Pre-built questions for testing
- **Session History**: Track all Q&A pairs
- **Dark/Light Mode**: Customizable appearance

### 🛠️ Production Ready
- **Modular Design**: Separated backend logic (`chatbot.py`) and frontend (`app.py`)
- **Error Handling**: Graceful fallbacks for edge cases
- **Performance Metrics**: Track response times and accuracy
- **Extensible**: Easy to add PDF, CSV, multi-file support

---

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    User Interface Layer                    │
│              (Streamlit Web App - app.py)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Chat Input  │  │  FAQ Editor  │  │  System Status  │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          │ Query + File Monitor
                          │
┌─────────────────────────▼─────────────────────────────────┐
│                 Backend Logic Layer                        │
│              (RAG Engine - chatbot.py)                     │
│                                                            │
│  ┌──────────────────────────────────────────────────┐    │
│  │        File Monitor & Event Trigger              │    │
│  │  • Detects faq.txt modifications                 │    │
│  │  • Triggers automatic re-indexing                │    │
│  └────────────────────┬─────────────────────────────┘    │
│                       │                                    │
│  ┌────────────────────▼─────────────────────────────┐    │
│  │           Document Processing                     │    │
│  │  • RecursiveCharacterTextSplitter                │    │
│  │  • Chunk size: 500, Overlap: 50                  │    │
│  └────────────────────┬─────────────────────────────┘    │
│                       │                                    │
│  ┌────────────────────▼─────────────────────────────┐    │
│  │           Embedding Generation                    │    │
│  │  • Model: sentence-transformers/all-MiniLM-L6-v2 │    │
│  │  • Converts text → 384-dim vectors               │    │
│  └────────────────────┬─────────────────────────────┘    │
│                       │                                    │
│  ┌────────────────────▼─────────────────────────────┐    │
│  │           Vector Store (FAISS)                    │    │
│  │  • Efficient similarity search                    │    │
│  │  • Top-K retrieval (default: 3 docs)             │    │
│  └────────────────────┬─────────────────────────────┘    │
│                       │                                    │
│  ┌────────────────────▼─────────────────────────────┐    │
│  │           LLM Answer Generation                   │    │
│  │  • Ollama: llama2, mistral, phi                  │    │
│  │  • OpenAI: GPT-3.5, GPT-4                        │    │
│  │  • Custom prompts with context injection         │    │
│  └────────────────────┬─────────────────────────────┘    │
│                       │                                    │
└───────────────────────┼────────────────────────────────────┘
                        │
                        │ Answer + Sources
                        │
┌───────────────────────▼────────────────────────────────────┐
│                   Response Layer                            │
│  • Formatted answer with citations                         │
│  • Source document snippets                                │
│  • Response time metrics                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- Python 3.10 or 3.11
- 4GB+ RAM (8GB recommended)
- Ollama installed OR OpenAI API key

### Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd live-rag-assistant

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull model
# Visit ollama.ai for installation
ollama pull llama2

# 5. Create FAQ file (or use default)
# faq.txt will be auto-created on first run

# 6. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

### Detailed Setup

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete instructions including:
- Step-by-step installation
- Configuration options
- Testing procedures
- Troubleshooting guide

---

## 🎯 Usage

### Basic Workflow

1. **Start Application**
```bash
streamlit run app.py
```

2. **Ask Questions**
- Type query in input box
- View answer with source citations
- Check response time metrics

3. **Update Knowledge Base**
- Edit `faq.txt` in any text editor
- Add/remove Q&A pairs
- Click "🔄 Refresh Vector Store"
- Ask questions about new content!

### Example Session

```python
# Initial FAQ content
Q: What is RAG?
A: RAG combines retrieval with generation...

# User asks: "What is RAG?"
# System returns answer from FAQ

# Developer adds to faq.txt:
Q: What is prompt engineering?
A: Prompt engineering is the art of crafting...

# Click refresh button

# User asks: "What is prompt engineering?"
# System returns answer from updated FAQ!
```

### Sample Queries

Try these in the app:
- "What is RAG?"
- "How does the live update feature work?"
- "Which LLM models are supported?"
- "What are embeddings?"
- "What is FAISS?"

---

## 🔧 Configuration

### Model Selection

**Ollama (Local)**
```python
chatbot = RAGChatbot(
    model_type="ollama",
    model_name="llama2"  # or "mistral", "phi"
)
```

**OpenAI (Cloud)**
```python
chatbot = RAGChatbot(
    model_type="openai",
    model_name="gpt-3.5-turbo"  # or "gpt-4"
)
```

### Advanced Parameters

```python
chatbot = RAGChatbot(
    faq_file="faq.txt",          # Knowledge base file
    model_type="ollama",          # "ollama" or "openai"
    model_name="llama2",          # Model name
    chunk_size=500,               # Text chunk size
    chunk_overlap=50,             # Chunk overlap
    top_k=3                       # Retrieved documents
)
```

### Performance Tuning

**For Speed:**
- Use smaller model: `phi` (1.3GB)
- Reduce chunk_size: `300`
- Reduce top_k: `2`

**For Accuracy:**
- Use larger model: `llama2` (7GB)
- Increase chunk_size: `700`
- Increase top_k: `5`

---

## 📁 Project Structure

```
live-rag-assistant/
│
├── app.py                 # Streamlit UI (frontend)
├── chatbot.py            # RAG engine (backend)
├── requirements.txt      # Python dependencies
├── faq.txt              # Knowledge base (auto-created)
│
├── README.md            # This file
├── SETUP_GUIDE.md       # Detailed setup instructions
│
├── venv/                # Virtual environment (gitignored)
└── __pycache__/         # Python cache (gitignored)
```

### File Descriptions

**app.py** (500+ lines)
- Streamlit UI components
- Chat interface
- FAQ editor
- System status dashboard
- Event handlers

**chatbot.py** (400+ lines)
- RAGChatbot class
- Document loading & processing
- Vector store management
- LLM integration
- File monitoring
- Query processing

**requirements.txt**
- All Python dependencies
- Version-pinned for stability

**faq.txt**
- Plain text FAQ content
- Q&A format
- Editable in any text editor

---

## 🧪 Testing

### Automated Tests

Create `test_chatbot.py`:

```python
from chatbot import RAGChatbot
import time

def test_initialization():
    """Test chatbot initialization"""
    chatbot = RAGChatbot(faq_file="faq.txt")
    assert chatbot.vector_store is not None
    print("✓ Initialization test passed")

def test_query():
    """Test query processing"""
    chatbot = RAGChatbot(faq_file="faq.txt")
    result = chatbot.query("What is RAG?")
    assert "retrieval" in result["answer"].lower()
    assert len(result["source_documents"]) > 0
    print("✓ Query test passed")

def test_live_update():
    """Test live FAQ updates"""
    chatbot = RAGChatbot(faq_file="faq.txt")
    
    # Add new content
    with open("faq.txt", "a") as f:
        f.write("\n\nQ: Test question?\nA: Test answer.")
    
    # Trigger update
    time.sleep(1)
    chatbot.check_for_updates()
    
    # Query new content
    result = chatbot.query("Test question")
    assert "test answer" in result["answer"].lower()
    print("✓ Live update test passed")

if __name__ == "__main__":
    test_initialization()
    test_query()
    test_live_update()
    print("\n✅ All tests passed!")
```

Run tests:
```bash
python test_chatbot.py
```

### Manual Testing

Test these scenarios:
1. Empty FAQ file
2. Very long FAQ (100+ entries)
3. Special characters in FAQ
4. Concurrent queries
5. Rapid FAQ updates
6. Large query text
7. Model switching

---

## 🚀 Deployment

### Local Network

```bash
streamlit run app.py --server.address 0.0.0.0
```
Access from other devices: `http://YOUR_IP:8501`

### Streamlit Cloud

1. Push to GitHub
2. Connect at share.streamlit.io
3. Deploy in 1 click

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

```bash
docker build -t rag-assistant .
docker run -p 8501:8501 rag-assistant
```

---

## 🎓 How It Works

### 1. Document Processing
```
FAQ Content → Text Splitter → Chunks (500 chars each)
```

### 2. Embedding Creation
```
Text Chunks → Embedding Model → 384-dim Vectors
```

### 3. Vector Storage
```
Vectors → FAISS Index → Fast Similarity Search
```

### 4. Query Processing
```
User Query → Embedding → Find Similar Vectors → Top 3 Chunks
```

### 5. Answer Generation
```
Query + Retrieved Context → LLM → Final Answer
```

### Live Update Mechanism

```python
# File monitoring
if file_modified_time > last_indexed_time:
    # Re-process entire FAQ
    load_and_index_documents()
    
    # Update vector store
    rebuild_faiss_index()
    
    # Update timestamp
    last_indexed_time = current_time
```

---

## 📊 Performance Metrics

### Typical Response Times
- **Query Processing**: 1-3 seconds
- **Vector Search**: <100ms
- **LLM Generation**: 1-2 seconds
- **FAQ Update**: 2-5 seconds

### Resource Usage
- **RAM**: 500MB-2GB (depends on model)
- **CPU**: Light usage
- **Storage**: <500MB total

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "Ollama not found"
```bash
# Solution: Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
```

**Issue**: Slow responses
```bash
# Solution: Use smaller model
ollama pull phi  # Only 1.3GB
```

**Issue**: Vector store not updating
```bash
# Solution: Manual refresh
# Click refresh button in UI
# Or restart app
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete troubleshooting guide.

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Multi-file support (PDF, DOCX, CSV)
- [ ] Chat history persistence
- [ ] User authentication
- [ ] Answer confidence scores
- [ ] Feedback mechanism
- [ ] Analytics dashboard
- [ ] API endpoints
- [ ] Mobile app

### Contribute
We welcome contributions! Areas for improvement:
1. Additional LLM providers
2. Better UI/UX
3. Advanced retrieval strategies
4. Performance optimizations
5. Documentation improvements

---

## 📚 Resources

### Documentation
- [LangChain Docs](https://docs.langchain.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Ollama Models](https://ollama.ai/library)

### Tutorials
- [RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [Embeddings Guide](https://huggingface.co/blog/getting-started-with-embeddings)

---

## 📄 License

MIT License - see LICENSE file for details

---

## 👥 Contributors

Created by [Your Name]

Special thanks to:
- LangChain team
- Streamlit team
- HuggingFace
- Meta AI (FAISS)
- Ollama

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Questions**: Check SETUP_GUIDE.md
- **Email**: your-email@example.com

---

## 🌟 Star This Project

If you find this useful, please give it a ⭐️ on GitHub!

---

**Built with ❤️ using LangChain, FAISS, and Streamlit**