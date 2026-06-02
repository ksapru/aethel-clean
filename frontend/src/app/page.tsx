"use client";
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import FileUploader from '../components/FileUploader';
import RiskDashboard from '../components/RiskDashboard';
import ChatInterface from '../components/ChatInterface';

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = async (files: FileList) => {
    setIsAnalyzing(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error(`API error: ${response.statusText}`);
      const data = await response.json();
      setAnalysisResult(data);
    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <main style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--canvas)', position: 'relative', overflowX: 'hidden' }}>
      <div className="hero-mesh animate-mesh"></div>
      
      {/* Header / Nav */}
      <header style={{ 
        padding: '32px 64px', 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        zIndex: 10
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
          <h1 style={{ fontSize: '1.4rem', fontWeight: 500, color: 'var(--ink)' }}>
            Aethel Intelligence
          </h1>
          <nav style={{ display: 'flex', gap: '24px', fontSize: '15px', color: 'var(--ink-secondary)' }}>
            <span>Portfolio</span>
            <span>Diligence</span>
            <span>Reports</span>
          </nav>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button className="button-primary-pill" style={{ fontSize: '14px' }}>Sign In</button>
        </div>
      </header>

      {/* Hero Section */}
      <section style={{ padding: '80px 64px 40px 64px', zIndex: 5 }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <span className="eyebrow">Institutional Secondary Intelligence</span>
          <h2 className="display-xl" style={{ maxWidth: '800px', marginBottom: '32px', color: 'var(--ink)' }}>
            The financial infrastructure for secondary fund diligence.
          </h2>
          <div style={{ display: 'flex', gap: '16px' }}>
             <button className="button-primary-pill" style={{ padding: '12px 24px' }}>Start Ingestion</button>
             <button className="button-secondary-pill" style={{ padding: '12px 24px' }}>View Sandbox</button>
          </div>
        </div>
      </section>

      <div className="dashboard-layout">
        {/* Left Side: Control & Status */}
        <aside className="dashboard-sidebar">
          <div className="stripi-card">
            <span className="eyebrow" style={{ color: 'var(--primary)' }}>Deal Ingestion</span>
            <FileUploader onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
          </div>

          <div className="stripi-card-dark">
            <span className="eyebrow" style={{ color: 'rgba(255,255,255,0.5)' }}>Specialist Swarm</span>
            <div style={{ display: 'grid', gap: '16px', marginTop: '16px' }}>
              {[
                { name: 'Orchestrator', status: 'Online' },
                { name: 'Liquidity Agent', status: 'Idle' },
                { name: 'Valuation Agent', status: 'Idle' },
                { name: 'Market Research', status: 'Idle' }
              ].map((agent, i) => (
                <div key={i} style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  fontSize: '14px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: agent.status === 'Online' ? '#533afd' : 'rgba(255,255,255,0.1)', border: agent.status === 'Online' ? '2px solid #fff' : 'none' }}></div>
                    <span style={{ fontWeight: 400 }}>{agent.name}</span>
                  </div>
                  <span style={{ fontSize: '11px', opacity: 0.5 }}>{agent.status}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* Right Side: Analysis & Chat */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '48px', minWidth: 0 }}>
          {analysisResult ? (
            <div className="memo-grid">
              {/* Primary Memo Card */}
              <div className="stripi-card" style={{ padding: '40px', boxSizing: 'border-box' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '40px' }}>
                  <div>
                    <span className="eyebrow">Executive Synthesis</span>
                    <h1 className="display-lg" style={{ margin: '8px 0 0 0' }}>Underwriting Memo</h1>
                  </div>
                  <button className="button-secondary-pill" style={{ fontSize: '13px' }}>Export PDF</button>
                </div>

                <div className="memo-content-area">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {analysisResult.ic_memo}
                  </ReactMarkdown>
                </div>
              </div>

              {/* Secondary Risk Card */}
              <div className="stripi-card risk-rail" style={{ padding: '32px', boxSizing: 'border-box' }}>
                 <RiskDashboard 
                  assessment={analysisResult.risk_analysis.overall_assessment} 
                  confidence={analysisResult.risk_analysis.confidence_score || 0.9}
                  keyDrivers={analysisResult.risk_analysis.key_drivers || []}
                  metrics={analysisResult.fund_summary}
                  dataGaps={analysisResult.risk_analysis.data_gaps || []}
                />
              </div>
            </div>
          ) : (
            <div className="stripi-card" style={{ height: '400px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', borderStyle: 'dashed' }}>
               <p style={{ color: 'var(--ink-mute)', fontSize: '15px' }}>Await document ingestion to initialize specialist swarm.</p>
            </div>
          )}

          <div className="stripi-card" style={{ padding: '48px' }}>
            <span className="eyebrow">Conversational Deal Room</span>
            <ChatInterface />
          </div>
        </main>
      </div>

      <footer style={{ padding: '64px', borderTop: '1px solid var(--hairline)', background: 'var(--canvas)', color: 'var(--ink-mute)', fontSize: '13px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between' }}>
           <span>© 2026 Aethel Intelligence Financial Infrastructure for Secondaries.</span>
           <div style={{ display: 'flex', gap: '32px' }}>
             <span>Security</span>
             <span>Terms</span>
             <span>Privacy</span>
           </div>
        </div>
      </footer>
    </main>
  );
}
