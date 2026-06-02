"use client";
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  agentName?: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentThought, setCurrentThought] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          history: messages
        })
      });

      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let currentMsg: Message = { role: 'assistant', content: '', agentName: '' };
      setMessages(prev => [...prev, currentMsg]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'agent_name') {
                currentMsg = { ...currentMsg, agentName: data.content };
              } else if (data.type === 'text') {
                currentMsg = { ...currentMsg, content: currentMsg.content + data.content };
                setCurrentThought(''); 
              } else if (data.type === 'thought') {
                setCurrentThought(data.content);
              } else if (data.type === 'error') {
                currentMsg = { ...currentMsg, content: currentMsg.content + `\n**Error:** ${data.content}` };
              }
              setMessages(prev => {
                const newArr = [...prev];
                newArr[newArr.length - 1] = currentMsg;
                return newArr;
              });
            } catch (e) {
              // Ignore partial JSON
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat failed", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "I encountered an error connecting to the agent swarm." }]);
    } finally {
      setLoading(false);
      setCurrentThought('');
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '600px', 
      width: '100%', 
      background: 'var(--canvas)'
    }}>
      {/* Message Area */}
      <div ref={scrollRef} style={{ 
        flex: 1, 
        overflowY: 'auto', 
        padding: '24px 0', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '32px' 
      }}>
        {messages.length === 0 && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', opacity: 0.3 }}>
             <p style={{ color: 'var(--ink-mute)', fontSize: '15px' }}>Enter a query to engage the swarm.</p>
          </div>
        )}
        
        {messages.map((msg, i) => (
          <div key={i} style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '8px',
            alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start'
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px',
              marginBottom: '4px'
            }}>
              {msg.role === 'assistant' && <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--primary)' }}></div>}
              <span className="eyebrow" style={{ marginBottom: 0, fontSize: '10px' }}>
                {msg.role === 'user' ? 'Principal Query' : (msg.agentName || 'Swarm Response')}
              </span>
            </div>
            <div className="markdown-content" style={{ 
              background: msg.role === 'user' ? 'var(--canvas-soft)' : 'transparent',
              padding: msg.role === 'user' ? '12px 20px' : '0',
              borderRadius: '18px',
              border: msg.role === 'user' ? '1px solid var(--hairline)' : 'none',
              fontSize: '15px',
              color: 'var(--ink-secondary)',
              maxWidth: '85%'
            }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--primary)', opacity: 0.5 }}></div>
            <span style={{ fontSize: '13px', color: 'var(--ink-mute)', fontStyle: 'italic' }}>
              {currentThought || 'Synthesizing...'}
            </span>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div style={{ padding: '24px 0', borderTop: '1px solid var(--hairline)' }}>
        <form onSubmit={handleSend} style={{ display: 'flex', gap: '12px' }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the deal..."
            style={{
              flex: 1,
              background: 'var(--canvas-soft)',
              border: '1px solid var(--hairline)',
              borderRadius: '9999px',
              padding: '12px 24px',
              color: 'var(--ink)',
              fontSize: '15px',
              outline: 'none'
            }}
          />
          <button 
            type="submit" 
            disabled={loading || !input.trim()}
            className="button-primary-pill"
            style={{ 
              opacity: loading || !input.trim() ? 0.5 : 1
            }}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
