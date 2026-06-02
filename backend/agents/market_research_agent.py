import os
import json
import time
from google import genai
from typing import Generator

class MarketResearchAgent:
    """Uses Google's Gemini Deep Research Agent via Interactions API."""
    
    def __init__(self):
        # Explicitly load the API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)
        
    def answer_query_stream(self, query: str, context: str = "") -> Generator[str, None, None]:
        prompt = f"""
        User Query: {query}
        
        Local Document Context:
        {context}
        
        Task:
        1. Analyze the local context if relevant.
        2. Conduct targeted web research to find supporting or missing data.
        3. Synthesize a concise, professional response. Focus on high-impact data points and multiples.
        Format your final response in clear Markdown. Keep it under 500 words to conserve token usage.
        """
        
        print(f"DEBUG: Starting Deep Research Task...")
        
        interaction_id = None
        last_event_id = None
        is_complete = False

        def process_stream(stream):
            nonlocal interaction_id, last_event_id, is_complete
            for event in stream:
                if event.event_id:
                    last_event_id = event.event_id
                
                if event.event_type == "interaction.created":
                    interaction_id = event.interaction.id
                    print(f"DEBUG: Interaction Created: {interaction_id}")
                
                if event.event_type == "step.delta":
                    if event.delta.type == "text":
                        data = json.dumps({"type": "text", "content": event.delta.text})
                        yield f"data: {data}\n\n"
                    elif event.delta.type == "thought":
                        print(f"DEBUG: Agent Thought: {event.delta.text[:50]}...")
                        data = json.dumps({"type": "thought", "content": event.delta.text})
                        yield f"data: {data}\n\n"
                elif event.event_type in ("interaction.completed", "error"):
                    print(f"DEBUG: Interaction {event.event_type}")
                    is_complete = True

        try:
            stream = self.client.interactions.create(
                input=prompt,
                agent="deep-research-preview-04-2026",
                background=True,
                stream=True,
                agent_config={"type": "deep-research", "thinking_summaries": "auto"},
            )
            
            yield from process_stream(stream)
            
            # Polling with backoff
            while not is_complete and interaction_id:
                time.sleep(4) # Respect rate limits and give the agent time to think
                
                status = self.client.interactions.get(interaction_id)
                if status.status != "in_progress":
                    # One last pull of the stream
                    final_stream = self.client.interactions.get(
                        id=interaction_id, stream=True, last_event_id=last_event_id
                    )
                    yield from process_stream(final_stream)
                    break
                
                # Get deltas since last event
                poll_stream = self.client.interactions.get(
                    id=interaction_id, stream=True, last_event_id=last_event_id
                )
                yield from process_stream(poll_stream)
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            error_data = json.dumps({"type": "error", "content": f"Research failed: {str(e)}"})
            yield f"data: {error_data}\n\n"
