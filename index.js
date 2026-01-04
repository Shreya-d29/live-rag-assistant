import React, { useState, useEffect, useRef } from 'react';
import { Send, FileText, RefreshCw, Clock, CheckCircle, AlertCircle } from 'lucide-react';

const LiveRAGAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [faqContent, setFaqContent] = useState('');
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [vectorStoreStatus, setVectorStoreStatus] = useState('ready');
  const messagesEndRef = useRef(null);

  // Sample FAQ content
  const sampleFAQ = `Q: What is RAG?
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
A: Typical response time is 1-3 seconds, depending on the LLM model, query complexity, and vector store size.`;

  useEffect(() => {
    setFaqContent(sampleFAQ);
    setMessages([{
      role: 'assistant',
      content: 'Hello! I\'m your Live RAG Assistant. I can answer questions based on the FAQ knowledge base. The system automatically updates when the FAQ content changes. Try asking me anything!',
      timestamp: new Date()
    }]);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Simulate vector store update
  const updateVectorStore = () => {
    setVectorStoreStatus('updating');
    setTimeout(() => {
      setLastUpdate(new Date());
      setVectorStoreStatus('ready');
      setMessages(prev => [...prev, {
        role: 'system',
        content: 'Vector store updated successfully with new FAQ content.',
        timestamp: new Date()
      }]);
    }, 2000);
  };

  // Simple retrieval simulation
  const retrieveRelevantChunks = (query) => {
    const lowerQuery = query.toLowerCase();
    const faqLines = faqContent.split('\n\n');
    
    // Score each FAQ entry
    const scored = faqLines.map(chunk => {
      const lowerChunk = chunk.toLowerCase();
      let score = 0;
      
      // Simple keyword matching
      const queryWords = lowerQuery.split(' ').filter(w => w.length > 3);
      queryWords.forEach(word => {
        if (lowerChunk.includes(word)) score += 1;
      });
      
      return { chunk, score };
    });
    
    // Return top 2 relevant chunks
    return scored
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 2)
      .map(item => item.chunk);
  };

  // Simulate LLM response generation
  const generateAnswer = async (query, context) => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    if (context.length === 0) {
      return "I couldn't find relevant information in the FAQ to answer your question. Please try rephrasing or ask about topics covered in the knowledge base.";
    }
    
    // Extract answer from context
    const answers = context.map(chunk => {
      const lines = chunk.split('\n');
      const answerLine = lines.find(line => line.startsWith('A:'));
      return answerLine ? answerLine.substring(2).trim() : '';
    }).filter(a => a);
    
    if (answers.length > 0) {
      return answers.join('\n\n') + '\n\n📚 Source: FAQ Knowledge Base';
    }
    
    return "Based on the available information, I found relevant content but couldn't extract a clear answer. Please check the FAQ for more details.";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Step 1: Retrieve relevant chunks
      const relevantChunks = retrieveRelevantChunks(input);
      
      // Step 2: Generate answer using LLM
      const answer = await generateAnswer(input, relevantChunks);
      
      const assistantMessage = {
        role: 'assistant',
        content: answer,
        timestamp: new Date(),
        sources: relevantChunks.length
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
                <FileText className="text-indigo-600" />
                Live RAG Assistant
              </h1>
              <p className="text-gray-600 mt-1">Real-Time FAQ & Knowledge Retrieval System</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Clock size={16} />
                  Last Update: {lastUpdate.toLocaleTimeString()}
                </div>
                <div className="flex items-center gap-2 text-sm mt-1">
                  {vectorStoreStatus === 'ready' ? (
                    <>
                      <CheckCircle size={16} className="text-green-600" />
                      <span className="text-green-600">Vector Store Ready</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw size={16} className="text-blue-600 animate-spin" />
                      <span className="text-blue-600">Updating...</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Chat Interface */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg flex flex-col" style={{ height: '600px' }}>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      msg.role === 'user'
                        ? 'bg-indigo-600 text-white'
                        : msg.role === 'system'
                        ? 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                    {msg.sources && (
                      <div className="text-xs mt-2 opacity-75">
                        Retrieved from {msg.sources} source{msg.sources > 1 ? 's' : ''}
                      </div>
                    )}
                    <div className="text-xs mt-2 opacity-75">
                      {msg.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg p-4">
                    <div className="flex items-center gap-2">
                      <RefreshCw size={16} className="animate-spin text-indigo-600" />
                      <span className="text-gray-600">Processing query...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
                  placeholder="Ask a question about the FAQ..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-600"
                  disabled={loading}
                />
                <button
                  onClick={handleSubmit}
                  disabled={loading || !input.trim()}
                  className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  <Send size={20} />
                  Send
                </button>
              </div>
            </div>
          </div>

          {/* FAQ Editor & System Info */}
          <div className="space-y-4">
            {/* FAQ Content */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-gray-800">FAQ Content (faq.txt)</h3>
                <button
                  onClick={updateVectorStore}
                  disabled={vectorStoreStatus === 'updating'}
                  className="px-3 py-1 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700 disabled:bg-gray-300 flex items-center gap-1"
                >
                  <RefreshCw size={14} />
                  Update
                </button>
              </div>
              <textarea
                value={faqContent}
                onChange={(e) => setFaqContent(e.target.value)}
                className="w-full h-64 p-3 border border-gray-300 rounded text-sm font-mono resize-none focus:outline-none focus:ring-2 focus:ring-indigo-600"
                placeholder="Add your FAQ content here..."
              />
              <p className="text-xs text-gray-500 mt-2">
                Edit the FAQ and click Update to rebuild the vector store
              </p>
            </div>

            {/* System Architecture */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="font-semibold text-gray-800 mb-3">RAG Pipeline</h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2 p-2 bg-blue-50 rounded">
                  <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                  <span>1. User Query</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-purple-50 rounded">
                  <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
                  <span>2. Retrieve Context (FAISS)</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-green-50 rounded">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span>3. Generate Answer (LLM)</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-indigo-50 rounded">
                  <div className="w-2 h-2 bg-indigo-600 rounded-full"></div>
                  <span>4. Display Response</span>
                </div>
              </div>
            </div>

            {/* Sample Queries */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="font-semibold text-gray-800 mb-3">Try These Queries</h3>
              <div className="space-y-2">
                {[
                  "What is RAG?",
                  "How does live update work?",
                  "Which LLM models are supported?",
                  "What are embeddings?"
                ].map((query, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInput(query)}
                    className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded transition-colors"
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveRAGAssistant;