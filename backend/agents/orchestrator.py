from typing import List, Dict, Any
import os
import re
from backend.services.parser import DocumentParser
from backend.services.rag_service import RAGService
from backend.agents.planner import DiligencePlanner
from backend.agents.specialists import LiquidityAgent, ValuationAgent, LeverageAgent
from backend.agents.synthesizer import DiligenceSynthesizer
from backend.agents.missing_diligence_agent import MissingDiligenceAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class InvestmentOrchestrator:
    """Refactored Orchestrator: Acts as the runtime for specialized agents."""

    def __init__(self):
        self.rag = RAGService()
        self.planner = DiligencePlanner()
        self.synthesizer = DiligenceSynthesizer()
        self.gap_agent = MissingDiligenceAgent()
        
        # Specialist map
        self.specialists = {
            "liquidity_agent": LiquidityAgent(),
            "valuation_agent": ValuationAgent(),
            "leverage_agent": LeverageAgent()
        }

        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def process_diligence(self, file_paths: List[str]) -> Dict[str, Any]:
        """Runs the modular agentic diligence pipeline with parallel parsing."""
        from concurrent.futures import ThreadPoolExecutor
        
        # 1. Ingest (Parallelized parsing)
        with ThreadPoolExecutor() as executor:
            parsed_docs = list(executor.map(DocumentParser.parse_document, file_paths))
            
        self.rag.add_documents(parsed_docs)
        
        # Extract source filters (basenames) to isolate RAG queries
        source_filters = [os.path.basename(fp) for fp in file_paths]
        
        # 2. Phase 1: Planning
        initial_context = self.rag.query_with_sources(
            "Overview of the fund, main strategy, and obvious risk factors.", 
            k=15, 
            source_filters=source_filters
        )
        plan = self.planner.generate_plan(initial_context)
        print(f"Plan Generated: {plan['rationale']}")
        
        # 3. Phase 2: Specialist Execution
        specialist_findings = {}
        for agent_id in plan.get("invoked_agents", []):
            if agent_id in self.specialists:
                print(f"Invoking {agent_id}...")
                # Query RAG for specialist-specific context
                query_map = {
                    "liquidity_agent": "Liquidity, unfunded commitments, fund life, exit environment.",
                    "valuation_agent": "Valuation methodology, NAV staleness, public multiples.",
                    "leverage_agent": "Debt, credit facilities, portfolio leverage."
                }
                spec_context = self.rag.query_with_sources(
                    query_map.get(agent_id, "General risk factors."),
                    source_filters=source_filters
                )
                specialist_findings[agent_id] = self.specialists[agent_id].analyze(spec_context)

        # 4. Phase 4: Gap Identification
        gaps = self.gap_agent.identify_gaps(initial_context)
        
        # 5. Metadata Extraction (Deterministic)
        metadata_context = self.rag.query_with_sources(
            "What is the fund's vintage year, investment strategy, total NAV, and TVPI performance?", 
            k=15, 
            source_filters=source_filters
        )
        metrics = self._extract_core_metrics(metadata_context)
        
        # 6. Phase 3: Synthesis
        ic_memo = self.synthesizer.synthesize(specialist_findings, metrics)


        # Format risks for UI compatibility
        ui_risks = []
        for agent_id, findings in specialist_findings.items():
            ui_risks.append({
                "type": agent_id.replace("_", " ").title(),
                "score": findings.get("score", "Medium"),
                "confidence": findings.get("confidence", 0.85),
                "evidence": findings.get("findings", "")
            })

        return {
            "fund_summary": metrics,
            "ic_memo": ic_memo,
            "risk_analysis": {
                "overall_assessment": plan.get("rationale", "Underwriting in progress."),
                "confidence_score": sum([f.get("confidence", 0) for f in specialist_findings.values()]) / len(specialist_findings) if specialist_findings else 0.85,
                "key_drivers": plan.get("critical_risks", []),
                "risks": ui_risks,
                "data_gaps": gaps
            },
            "sources": [doc["file_name"] for doc in parsed_docs]
        }

    def _extract_core_metrics(self, context: str) -> Dict[str, Any]:
        """Extracts fund performance metrics using deterministic regex + LLM fallback."""
        vintage_match = re.search(r"(?:Vintage|Year):\s*(\d{4})", context, re.IGNORECASE)
        strategy_match = re.search(r"Strategy:\s*([^\n\.,]+)", context, re.IGNORECASE)
        nav_match = re.search(r"NAV:\s*([\$€£]?[\d\.,]+[MBT]?)", context, re.IGNORECASE)
        tvpi_match = re.search(r"TVPI:\s*([\d\.]+x?)", context, re.IGNORECASE)

        prompt = ChatPromptTemplate.from_template("""
        Extract fund metrics from context. JSON only.
        
        Output:
        {{
            "vintage": "...",
            "strategy": "...",
            "nav": "...",
            "tvpi": "...",
            "purchase_price": "...",
            "implied_entry": "...",
            "liquidity_profile": "..."
        }}
        Context: {context}
        """)
        chain = prompt | self.llm
        response = chain.invoke({"context": context})
        import json
        try:
            metrics = json.loads(response.content.replace("```json", "").replace("```", "").strip())
            if vintage_match: metrics["vintage"] = vintage_match.group(1)
            if strategy_match: metrics["strategy"] = strategy_match.group(1).strip()
            if nav_match: metrics["nav"] = nav_match.group(1).strip()
            if tvpi_match: metrics["tvpi"] = tvpi_match.group(1)
            return metrics
        except:
            return {"error": "extraction failed"}
