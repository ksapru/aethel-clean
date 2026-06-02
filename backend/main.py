import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict
import shutil

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PE Secondary Diligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None
_chat_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from backend.agents.orchestrator import InvestmentOrchestrator
        _orchestrator = InvestmentOrchestrator()
    return _orchestrator

def get_chat_orchestrator():
    global _chat_orchestrator
    if _chat_orchestrator is None:
        from backend.agents.chat_orchestrator import ChatOrchestrator
        _chat_orchestrator = ChatOrchestrator()
    return _chat_orchestrator

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./backend/data/uploads")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

@app.post("/analyze")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """Uploads documents and triggers the agentic RAG pipeline."""
    file_paths = []
    try:
        for file in files:
            path = os.path.join(UPLOAD_DIR, file.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(path)

        orchestrator = get_orchestrator()
        result = orchestrator.process_diligence(file_paths)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_agents(request: ChatRequest):
    """Interactive chat with the agent swarm."""
    try:
        chat_orchestrator = get_chat_orchestrator()
        result = chat_orchestrator.answer_query(request.query, request.history)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """Interactive chat with the agent swarm, supporting streaming."""
    try:
        chat_orchestrator = get_chat_orchestrator()
        return StreamingResponse(
            chat_orchestrator.answer_query_stream(request.query, request.history),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"CRITICAL ERROR in chat_stream: {e}")
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
