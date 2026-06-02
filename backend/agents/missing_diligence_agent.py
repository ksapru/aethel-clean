import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class MissingDiligenceAgent:
    """Phase 4 Agent: Identifies critical data gaps."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def identify_gaps(self, context: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template("""
        You are a Diligence Auditor. Based on the provided context, identify what CRITICAL information is missing for a standard PE secondary underwriting.
        Look for:
        - Missing specific quarterly reports
        - Lack of underlying portfolio company details
        - Missing unfunded commitment schedules
        - Incomplete GP-led transaction terms
        
        Context: {context}
        
        Output: A simple JSON list of strings (data gaps).
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        
        import json
        try:
            return json.loads(response.content.replace("```json", "").replace("```", "").strip())
        except:
            return ["Review of full data room recommended for gap identification."]
