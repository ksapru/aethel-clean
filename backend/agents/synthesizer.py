import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class DiligenceSynthesizer:
    """Phase 3 Agent: Combines all outputs into a final IC Memo."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def synthesize(self, specialist_findings: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        prompt = ChatPromptTemplate.from_template("""
        You are the Investment Committee Chair at Aethel Intelligence. Synthesize the findings from our specialist agents into a final IC Memo.
        
        SPECIALIST FINDINGS:
        {findings}
        
        FUND METRICS:
        {metrics}
        
        Provide:
        1. RECOMMENDATION: Clear Buy/Hold/Pass.
        2. EXECUTIVE SUMMARY: High-level thesis.
        3. KEY DRIVERS: Top 3-5 drivers.
        4. CONCLUSION.
        
        Format in professional GitHub Markdown.
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({
            "findings": specialist_findings,
            "metrics": metrics
        })
        
        return response.content
