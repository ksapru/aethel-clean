"use client";
import React, { useState, useRef } from 'react';

interface FileUploaderProps {
  onAnalyze: (files: FileList) => void;
  isAnalyzing: boolean;
}

export default function FileUploader({ onAnalyze, isAnalyzing }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (newFiles: FileList) => {
    const fileList = Array.from(newFiles);
    setFiles(prev => [...prev, ...fileList]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const uploadFiles = () => {
    const dataTransfer = new DataTransfer();
    files.forEach(file => dataTransfer.items.add(file));
    onAnalyze(dataTransfer.files);
  };

  return (
    <div style={{ display: 'grid', gap: '24px' }}>
      <div 
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        style={{
          border: '1px dashed var(--hairline)',
          borderRadius: '12px',
          padding: '32px 20px',
          textAlign: 'center',
          background: isDragging ? 'var(--canvas-soft)' : 'transparent',
          transition: 'all 0.2s',
          cursor: 'pointer'
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          multiple 
          ref={fileInputRef} 
          onChange={(e) => e.target.files && handleFiles(e.target.files)} 
          style={{ display: 'none' }} 
        />
        <div style={{ fontSize: '24px', marginBottom: '12px', color: 'var(--primary)', opacity: 0.8 }}>↑</div>
        <p style={{ fontSize: '14px', color: 'var(--ink-secondary)', fontWeight: 300 }}>
          {isDragging ? 'Drop to upload' : 'Click or drop deal docs'}
        </p>
      </div>

      {files.length > 0 && (
        <div style={{ display: 'grid', gap: '12px' }}>
          {files.map((f, i) => (
            <div key={i} style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              fontSize: '13px',
              color: 'var(--ink-secondary)',
              padding: '8px 12px',
              background: 'var(--canvas-soft)',
              borderRadius: '8px',
              border: '1px solid var(--hairline)'
            }}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '200px' }}>{f.name}</span>
              <button 
                onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                style={{ background: 'none', border: 'none', color: 'var(--ink-mute)', cursor: 'pointer', fontSize: '16px' }}
              >
                ×
              </button>
            </div>
          ))}
          
          <button 
            className="button-primary-pill" 
            onClick={uploadFiles} 
            disabled={isAnalyzing}
            style={{ width: '100%', marginTop: '8px', opacity: isAnalyzing ? 0.5 : 1 }}
          >
            {isAnalyzing ? 'Initializing Swarm...' : 'Analyze Documents'}
          </button>
        </div>
      )}
    </div>
  );
}
