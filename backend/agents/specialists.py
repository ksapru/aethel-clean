import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class BaseSpecialist:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

class LiquidityAgent(BaseSpecialist):
    def __init__(self):
        super().__init__("Liquidity Agent")

    def analyze(self, context: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template("""
        You are a Liquidity Specialist. Analyze the context for:
        - Unfunded commitments
        - Fund life and extension options
        - GP-led transaction volume and exit expectations
        
        Context: {context}
        
        Output JSON:
        {{
            "score": "High|Medium|Low",
            "findings": "...",
            "evidence": ["...", "..."],
            "confidence": 0.0-1.0
        }}
        """)
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        import json
        return json.loads(response.content.replace("```json", "").replace("```", "").strip())

class ValuationAgent(BaseSpecialist):
    def __init__(self):
        super().__init__("Valuation Agent")

    def analyze(self, context: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template("""
        You are a Valuation Specialist. Analyze the context for:
        - NAV reporting frequency and staleness
        - Public market comparable multiples
        - Discounted cash flow assumptions
        
        Context: {context}
        
        Output JSON:
        {{
            "score": "High|Medium|Low",
            "findings": "...",
            "evidence": ["...", "..."],
            "confidence": 0.0-1.0
        }}
        """)
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        import json
        return json.loads(response.content.replace("```json", "").replace("```", "").strip())

class LeverageAgent(BaseSpecialist):
    def __init__(self):
        super().__init__("Leverage Agent")

    def analyze(self, context: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template("""
        You are a Leverage Specialist. Analyze the context for:
        - Fund-level credit facilities (subscription lines)
        - Portfolio company debt-to-EBITDA ratios
        - Interest rate sensitivity
        
        Context: {context}
        
        Output JSON:
        {{
            "score": "High|Medium|Low",
            "findings": "...",
            "evidence": ["...", "..."],
            "confidence": 0.0-1.0
        }}
        """)
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        import json
        return json.loads(response.content.replace("```json", "").replace("```", "").strip())
