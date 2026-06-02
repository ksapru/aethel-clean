from typing import List, Dict, Any
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class RiskEngine:
    """Evaluates investment risk based on retrieved fund data."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def evaluate_risk(self, context: str) -> Dict[str, Any]:
        """Scores specific risk vectors based on provided document context."""
        
        prompt = ChatPromptTemplate.from_template("""
        You are a Senior Risk Analyst for a PE Secondary Fund.
        Based ON ONLY the provided document context, evaluate the following risk vectors.
        Assign a Qualitative Score (High, Medium, Low) and provide a concise 'Evidence' summary.
        Include a source citation for the evidence in parentheses, e.g., (document_name.pdf).

        Risk Vectors:
        1. Liquidity Risk (Exit environment, unfunded commitments, fund life)
        2. Valuation Risk (NAV staleness, public multiple compression, Level 3 assets)
        3. Leverage Risk (Portfolio company debt, fund-level credit lines)
        4. Operational Risk (GP stability, strategy drift, reporting quality)

        Context:
        {context}

        Output format (JSON):
        {{
            "overall_assessment": "...",
            "confidence_score": 0.0-1.0,
            "key_drivers": ["...", "..."],
            "risks": [
                {{ "type": "Liquidity Risk", "score": "High|Medium|Low", "confidence": 0.0-1.0, "evidence": "Summary (source_file.pdf)" }},
                {{ "type": "Valuation Risk", "score": "High|Medium|Low", "confidence": 0.0-1.0, "evidence": "Summary (source_file.pdf)" }},
                {{ "type": "Leverage Risk", "score": "High|Medium|Low", "confidence": 0.0-1.0, "evidence": "Summary (source_file.pdf)" }},
                {{ "type": "Operational Risk", "score": "High|Medium|Low", "confidence": 0.0-1.0, "evidence": "Summary (source_file.pdf)" }}
            ],
            "red_flags": ["...", "..."]
        }}
        """)

        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        
        import json
        try:
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            return {
                "overall_assessment": "Error extracting risk signals.",
                "confidence_score": 0.0,
                "key_drivers": ["Extraction failure"],
                "risks": [
                    { "type": "Liquidity Risk", "score": "Unknown", "confidence": 0, "evidence": "N/A" },
                    { "type": "Valuation Risk", "score": "Unknown", "confidence": 0, "evidence": "N/A" },
                    { "type": "Leverage Risk", "score": "Unknown", "confidence": 0, "evidence": "N/A" },
                    { "type": "Operational Risk", "score": "Unknown", "confidence": 0, "evidence": "N/A" }
                ],
                "red_flags": [str(e)]
            }
