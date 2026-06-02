import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class DiligencePlanner:
    """Phase 1 Agent: Decides which risks to analyze and agents to invoke."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_plan(self, initial_context: str) -> Dict[str, Any]:
        """Analyzes context and outputs a structured execution plan."""
        
        prompt = ChatPromptTemplate.from_template("""
        You are the Head of Underwriting for a Multi-Billion Dollar Secondary Fund. 
        Given the initial document context, create an Executive Underwriting Assessment.
        
        Context Snippets:
        {context}
        
        Specialist Agents Available:
        - liquidity_agent: Analyzes unfunded commitments, fund life, and exit environments.
        - valuation_agent: Analyzes NAV methodology, multiple compression, and Level 3 assets.
        - leverage_agent: Analyzes fund-level and portfolio-level debt.
        
        Output format (JSON):
        {{
            "critical_risks": ["Risk A", "Risk B"],
            "invoked_agents": ["agent_name_1", "agent_name_2"],
            "rationale": "A highly professional, institutional Executive Assessment. Summarize the deal's core investment thesis and why specific risk vectors (Liquidity, Valuation, or Leverage) are being prioritized. DO NOT mention internal agent names like 'liquidity_agent'. Use phrases like 'Priority analysis is required on...' or 'The underwriting focus centers on...'."
        }}
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"context": initial_context})
        
        import json
        try:
            return json.loads(response.content.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            return {
                "critical_risks": ["Liquidity", "Valuation", "Leverage"],
                "invoked_agents": ["liquidity_agent", "valuation_agent", "leverage_agent"],
                "rationale": f"Fallback due to planner error: {str(e)}"
            }
