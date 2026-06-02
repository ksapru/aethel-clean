import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.services.rag_service import RAGService
from backend.agents.market_research_agent import MarketResearchAgent
import json
from typing import Generator
class ChatOrchestrator:
    """Agentic Chat: Routes user queries to specialists or general RAG."""
    
    def __init__(self):
        self.rag = RAGService()
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Note: We keep the k=2 context limit in answer_query_stream to save cost

    def answer_query(self, query: str, history: List[Dict[str, str]] = []) -> Dict[str, str]:
        """Answers follow-up questions about the deal documents."""
        
        # Implementing query splitting to ensure multi-part questions get diverse context
        import re
        sub_queries = re.split(r'[.!?]\s+', query)
        if len(sub_queries) > 1:
            all_context = ""
            for sq in sub_queries:
                if len(sq.strip()) > 10:
                    all_context += self.rag.query_with_sources(sq, k=8)
            context = all_context
        else:
            context = self.rag.query_with_sources(query, k=12)
        
        # 2. Determine agent persona (Routing)
        routing_prompt = ChatPromptTemplate.from_template("""
        Given the user query, which Aethel Intelligence specialist is best suited to answer?
        - Liquidity Agent: For unfunded commitments, cash flows, exits.
        - Valuation Agent: For NAV, multiples, fair value.
        - Leverage Agent: For debt, interest, credit.
        - Diligence Auditor: For missing data or compliance.
        - Market Research Agent: For external, live macroeconomic data, public comps, or news.
        
        Query: {query}
        
        Return ONLY the agent name.
        """)
        
        routing_chain = routing_prompt | self.llm
        agent_name = routing_chain.invoke({"query": query}).content.strip()
        
        # 3. Generate response with persona
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are the {agent_name} at Aethel Intelligence. Answer the user's question using the provided document context. Be precise, institutional, and cite your sources."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])
        
        # Convert history to LangChain format
        from langchain_core.messages import HumanMessage, AIMessage
        lc_history = []
        for msg in history:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            else:
                lc_history.append(AIMessage(content=msg["content"]))

        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": query,
            "history": lc_history
        })
        
        return {
            "response": response.content,
            "agent_name": agent_name
        }

    def answer_query_stream(self, query: str, history: List[Dict[str, str]] = []) -> Generator[str, None, None]:
        try:
            # 1. Route query
            routing_prompt = ChatPromptTemplate.from_template("""
            Given the user query, which Aethel Intelligence specialist is best suited to answer?
            - Liquidity Agent: For unfunded commitments, cash flows, exits.
            - Valuation Agent: For NAV, multiples, fair value.
            - Leverage Agent: For debt, interest, credit.
            - Diligence Auditor: For missing data or compliance.
            - Market Research Agent: For external, live macroeconomic data, public comps, or news.
            
            Query: {query}
            
            Return ONLY the agent name.
            """)
            routing_chain = routing_prompt | self.llm
            agent_name = routing_chain.invoke({"query": query}).content.strip()
            
            # Send agent name immediately
            yield f"data: {json.dumps({'type': 'agent_name', 'content': agent_name})}\n\n"
            
            # Implementing query splitting to handle multi-part diligence questions
            # We split the query by sentence/question and retrieve context for each to avoid "Salesforce distraction"
            import re
            sub_queries = re.split(r'[.!?]\s+', query)
            if len(sub_queries) > 1:
                yield f"data: {json.dumps({'type': 'thought', 'content': f'Decomposing query into {len(sub_queries)} sub-tasks for deep search...'})}\n\n"
                all_context = ""
                for sq in sub_queries:
                    if len(sq.strip()) > 10:
                        all_context += self.rag.query_with_sources(sq, k=8)
                context = all_context
            else:
                yield f"data: {json.dumps({'type': 'thought', 'content': 'Retrieving comprehensive deal context (k=12)...'})}\n\n"
                context = self.rag.query_with_sources(query, k=12)
            
            if agent_name == "Market Research Agent":
                research_agent = MarketResearchAgent()
                yield from research_agent.answer_query_stream(query, context)
                return
            
            yield f"data: {json.dumps({'type': 'thought', 'content': f'Synthesizing specialist response as {agent_name}...'})}\n\n"
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are the {agent_name} at Aethel Intelligence. Answer the user's question using the provided document context. IMPORTANT: If you see context from unrelated public companies (like Salesforce or Nvidia), prioritize the Private Equity fund documents (Letters, Continuation Reports) first. Be precise, institutional, and cite your sources."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context:\n{context}\n\nQuestion: {query}")
            ])
            
            from langchain_core.messages import HumanMessage, AIMessage
            lc_history = []
            for msg in history:
                if msg["role"] == "user":
                    lc_history.append(HumanMessage(content=msg["content"]))
                else:
                    lc_history.append(AIMessage(content=msg["content"]))

            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "query": query,
                "history": lc_history
            })
            
            yield f"data: {json.dumps({'type': 'text', 'content': response.content})}\n\n"
        except Exception as e:
            print(f"ERROR in Orchestrator Stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
