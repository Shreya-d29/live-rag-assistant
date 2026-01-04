import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import Document

try:
    from langchain_community.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGChatbot:
    """
    Live RAG Assistant with automatic FAQ updates
    
    Features:
    - Real-time knowledge base updates
    - Vector store management with FAISS
    - Support for Ollama and OpenAI models
    - Document retrieval and answer generation
    """
    
    def __init__(
        self,
        faq_file: str = "faq.txt",
        model_type: str = "ollama",
        model_name: str = "llama2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3
    ):
        """
        Initialize the RAG Chatbot
        
        Args:
            faq_file: Path to the FAQ text file
            model_type: Type of LLM ("ollama" or "openai")
            model_name: Name of the model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            top_k: Number of relevant documents to retrieve
        """
        self.faq_file = faq_file
        self.model_type = model_type
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Track file modification time for auto-updates
        self.last_modified = None
        
        # Initialize components
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        # Setup
        self._initialize_embeddings()
        self.initialize_llm()
        self.load_and_index_documents()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embeddings model loaded")
    
    def initialize_llm(self):
        """Initialize the LLM based on model_type"""
        print(f"Initializing {self.model_type} LLM: {self.model_name}...")
        
        if self.model_type == "ollama":
            self.llm = Ollama(
                model=self.model_name,
                temperature=0.7
            )
        elif self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI is not installed. Run: pip install openai langchain-openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.7,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        print(f"✓ {self.model_type} LLM initialized")
    
    def load_and_index_documents(self) -> bool:
        """
        Load FAQ file, split into chunks, create embeddings, and build vector store
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            faq_path = Path(self.faq_file)
            
            if not faq_path.exists():
                print(f"Warning: FAQ file '{self.faq_file}' not found")
                return False
            
            # Update last modified time
            self.last_modified = faq_path.stat().st_mtime
            
            print(f"Loading FAQ from: {self.faq_file}")
            
            # Read FAQ content
            with open(self.faq_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("Warning: FAQ file is empty")
                return False
            
            # Create Document object
            documents = [Document(page_content=content, metadata={"source": self.faq_file})]
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            print(f"✓ Split into {len(chunks)} chunks")
            
            # Create vector store
            if self.vector_store is None:
                print("Creating new vector store...")
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                print("Updating existing vector store...")
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            print(f"✓ Vector store created with {len(chunks)} documents")
            
            # Create QA chain
            self._create_qa_chain()
            
            return True
            
        except Exception as e:
            print(f"Error loading and indexing documents: {str(e)}")
            return False
    
    def _create_qa_chain(self):
        """Create the question-answering chain"""
        
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant answering questions based on the provided context from an FAQ knowledge base.

Use the following pieces of context to answer the question at the end. If you don't know the answer or if the context doesn't contain relevant information, just say that you don't have enough information to answer accurately. Don't try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("✓ QA chain created")
    
    def check_for_updates(self) -> bool:
        """
        Check if FAQ file has been modified since last load
        
        Returns:
            bool: True if file was modified and reloaded, False otherwise
        """
        try:
            faq_path = Path(self.faq_file)
            
            if not faq_path.exists():
                return False
            
            current_mtime = faq_path.stat().st_mtime
            
            if self.last_modified is None or current_mtime > self.last_modified:
                print(f"FAQ file updated. Reloading...")
                return self.load_and_index_documents()
            
            return False
            
        except Exception as e:
            print(f"Error checking for updates: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            
        Returns:
            Dict containing answer and source documents
        """
        # Check for updates before querying
        self.check_for_updates()
        
        if self.qa_chain is None:
            return {
                "answer": "System not initialized. Please check if FAQ file exists.",
                "source_documents": []
            }
        
        try:
            print(f"\nQuery: {question}")
            start_time = time.time()
            
            # Get response from QA chain
            result = self.qa_chain.invoke({"query": question})
            
            elapsed_time = time.time() - start_time
            print(f"Response time: {elapsed_time:.2f}s")
            
            # Extract source texts
            source_docs = []
            if "source_documents" in result:
                source_docs = [doc.page_content for doc in result["source_documents"]]
            
            return {
                "answer": result["result"],
                "source_documents": source_docs,
                "response_time": elapsed_time
            }
            
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": []
            }
    
    def get_vector_store_status(self) -> Dict:
        """
        Get current status of the vector store
        
        Returns:
            Dict with status information
        """
        if self.vector_store is None:
            return {
                "initialized": False,
                "num_chunks": 0
            }
        
        try:
            # Get number of documents in vector store
            num_docs = self.vector_store.index.ntotal
            
            return {
                "initialized": True,
                "num_chunks": num_docs,
                "last_modified": datetime.fromtimestamp(self.last_modified) if self.last_modified else None
            }
        except Exception as e:
            return {
                "initialized": False,
                "num_chunks": 0,
                "error": str(e)
            }
    
    def add_documents(self, documents: List[str]) -> bool:
        """
        Add new documents to the vector store
        
        Args:
            documents: List of text documents to add
            
        Returns:
            bool: True if successful
        """
        try:
            if not documents:
                return False
            
            # Create Document objects
            docs = [Document(page_content=doc, metadata={"source": "manual_add"}) 
                   for doc in documents]
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(docs)
            
            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            
            print(f"✓ Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    print("=== RAG Chatbot Test ===\n")
    
    # Create chatbot instance
    chatbot = RAGChatbot(
        faq_file="faq.txt",
        model_type="ollama",
        model_name="llama2"
    )
    
    # Test queries
    test_queries = [
        "What is RAG?",
        "How does the live update feature work?",
        "Which models are supported?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        result = chatbot.query(query)
        print(f"Q: {query}")
        print(f"A: {result['answer']}")
        print(f"\nSources used: {len(result['source_documents'])}")
        print(f"Response time: {result.get('response_time', 0):.2f}s")
    
    # Check status
    print(f"\n{'='*60}")
    print("Vector Store Status:")
    status = chatbot.get_vector_store_status()
    print(f"  Initialized: {status['initialized']}")
    print(f"  Number of chunks: {status['num_chunks']}")