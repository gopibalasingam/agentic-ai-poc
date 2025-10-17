"""  
Research Agent Reference Implementation  
=======================================  
A modular AI research assistant with memory and tool(s) to support web search capabilities  
  
Modules:  
1. Configuration Manager - Centralized configuration settings  
2. Memory System - Conversation and search history management  
3. LLM Core - Base AI model interactions  
4. Tool Layer - Web search functionality  
5. Agent Components - Planning/Extraction/Synthesis logic  
6. Orchestrator - Main execution flow

Python libaries to install:
pip install openai
pip install serpapi
pip install google-search-results
"""  
  
# Standard library imports  
import json  
import re  
import time  
from typing import List, Dict, Any, Deque, Tuple  
from collections import deque  
from functools import lru_cache  
  
# Third-party imports  
from openai import AzureOpenAI  
from serpapi import GoogleSearch  
  
# ========================  
# 1. Configuration Manager  
# ========================  
class Config:  
    """Central configuration for API credentials and settings"""  
    AZURE_ENDPOINT = "<end point>"  
    AZURE_API_KEY = "<api key>"  
    AZURE_MODEL = "gpt-4.1"  
    SERPAPI_KEY = "<key>"  
    MAX_HISTORY = 5  # Number of interactions to remember  
    SEARCH_CACHE_SIZE = 32  # LRU cache size for search results  
  
# Initialize Azure client once at startup  
_azure_client = AzureOpenAI(  
    azure_endpoint=Config.AZURE_ENDPOINT,  
    api_key=Config.AZURE_API_KEY,  
    api_version="2025-01-01-preview"  
)  
  
# =================  
# 2. Memory System  
# =================  
class AgentMemory:  
    """  
    Maintains context for the agent including:  
    - Conversation history  
    - Previous search results  
      
    Attributes:  
        conversation: Deque of (role, content) tuples  
        searches: Deque of search result dictionaries  
    """  
      
    def __init__(self):  
        self.conversation: Deque[Tuple[str, str]] = deque(maxlen=Config.MAX_HISTORY)  
        self.searches: Deque[Dict[str, Any]] = deque(maxlen=Config.MAX_HISTORY)  
  
    def add_interaction(self, user_input: str, agent_response: str) -> None:  
        """Record a user-assistant interaction pair"""  
        self.conversation.extend([  
            ("user", user_input),  
            ("assistant", agent_response)  
        ])  
  
    def add_search(self, query: str, results: List[Dict[str, Any]]) -> None:  
        """Store search query and its results"""  
        self.searches.append({  
            "timestamp": time.time(),  
            "query": query,  
            "results": results  
        })  
  
    def get_context(self) -> str:  
        """Generate formatted context string for LLM prompts"""  
        context = ["Conversation History:"]  
        context.extend(f"{role.upper()}: {content}" for role, content in self.conversation)  
          
        context.append("\nRecent Searches:")  
        context.extend(  
            f"  {i}. {s['query']} ({len(s['results'])} results)"  
            for i, s in enumerate(self.searches, 1)  
        )  
          
        return "\n".join(context)  
  
# ==============  
# 3. LLM Core  
# ==============  
class LLMHandler:  
    """Handles interactions with Azure OpenAI models"""  
      
    @staticmethod  
    def generate_response(messages: List[Dict[str, str]],   
                         temperature: float = 0.0,  
                         max_tokens: int = 1000) -> str:  
        """  
        Execute chat completion request  
          
        Args:  
            messages: List of message dictionaries  
            temperature: Creativity parameter (0-1)  
            max_tokens: Maximum response length  
              
        Returns:  
            Generated text content  
              
        Raises:  
            RuntimeError: On API failure  
        """  
        try:  
            response = _azure_client.chat.completions.create(  
                model=Config.AZURE_MODEL,  
                messages=messages,  
                temperature=temperature,  
                max_tokens=max_tokens,  
                timeout=30  
            )  
            return response.choices[0].message.content  
        except Exception as e:  
            raise RuntimeError(f"LLM API Error: {str(e)}")  
  
    @staticmethod  
    def parse_json_response(text: str) -> Any:  
        """  
        Extract first valid JSON from text with fallback parsing  
          
        Args:  
            text: Text potentially containing JSON  
              
        Returns:  
            Parsed JSON object  
              
        Raises:  
            ValueError: If no valid JSON found  
        """  
        try:  
            return json.loads(text)  
        except json.JSONDecodeError:  
            pass  
          
        # Fallback nested JSON detection  
        matches = re.findall(r'\{(?:[^{}]|(?R))*\}', text)  
        for match in matches:  
            try:  
                return json.loads(match)  
            except json.JSONDecodeError:  
                continue  
                  
        raise ValueError("No valid JSON found in response")  
  
# ================  
# 4. Search Tool  
# ================  
class WebSearch:  
    """Handles search operations with caching and result parsing"""  
      
    @classmethod  
    @lru_cache(maxsize=Config.SEARCH_CACHE_SIZE)  
    def execute(cls, query: str, count: int = 5) -> List[Dict[str, Any]]:  
        """  
        Perform web search with result caching  
          
        Args:  
            query: Search query string  
            count: Number of results to return  
              
        Returns:  
            List of search result dictionaries  
              
        Raises:  
            RuntimeError: On search API failure  
        """  
        params = {  
            "q": query,  
            "num": count,  
            "api_key": Config.SERPAPI_KEY  
        }  
          
        try:  
            results = GoogleSearch(params).get_dict()  
            return cls._parse_results(results, count)  
        except Exception as e:  
            raise RuntimeError(f"Search failed: {str(e)}")  
  
    @staticmethod  
    def _parse_results(data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:  
        """Parse raw API response into standardized format"""  
        output = []  
        sources = [  
            *data.get("organic_results", []),  
            *data.get("top_results", []),  
            *(data.get("knowledge_graph", {}).get("sources", []) or []),  
            data.get("answer_box", {})  
        ]  
          
        for item in sources:  
            if len(output) >= max_results:  
                break  
                  
            result = {  
                "name": item.get("title") or item.get("name") or "",  
                "url": item.get("link") or item.get("url") or "",  
                "snippet": item.get("snippet") or item.get("description") or ""  
            }  
              
            if result["url"]:  
                output.append(result)  
          
        return output[:max_results]  
  
# ====================  
# 5. Agent Components  
# ====================  
class ResearchPlanner:  
    """Generates research plans based on current context"""  
      
    @staticmethod  
    def create_plan(question: str, memory: AgentMemory) -> Dict[str, Any]:  
        """  
        Generate a JSON research plan  
          
        Args:  
            question: User's research question  
            memory: Current agent memory  
              
        Returns:  
            Dictionary with 'steps' list of action objects  
        """  
        system_msg = f"""You are a research planner. Use this context:  
{memory.get_context()}  
  
Available tools:  
- web_search(query): Find current web information  
  
Output JSON plan with "steps" array containing action objects."""  
          
        response = LLMHandler.generate_response([  
            {"role": "system", "content": system_msg},  
            {"role": "user", "content": f"Question: {question}\nOutput plan JSON:"}  
        ])  
          
        return LLMHandler.parse_json_response(response)  
  
class FactExtractor:  
    """Handles information extraction from search results"""  
      
    @staticmethod  
    def process_results(question: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:  
        """  
        Extract structured facts from search results  
          
        Args:  
            question: Original research question  
            results: List of search results  
              
        Returns:  
            Dictionary with answer, confidence, and sources  
        """  
        formatted_results = "\n".join(  
            f"{i}. {r['name']}\n   {r['url']}\n   {r['snippet']}"  
            for i, r in enumerate(results, 1)  
        )  
          
        response = LLMHandler.generate_response([  
            {"role": "system", "content": """Extract facts from results. Output JSON:  
{"answer": "...", "confidence": 0.0-1.0, "sources": [{"url": "...", ...}]}"""},  
            {"role": "user", "content": f"Question: {question}\nResults:\n{formatted_results}"}  
        ])  
          
        extracted = LLMHandler.parse_json_response(response)  
        return FactExtractor._validate_sources(extracted, results)  
  
    @staticmethod  
    def _validate_sources(data: Dict, results: List[Dict]) -> Dict:  
        """Ensure sources match actual search results"""  
        valid_sources = []  
        for source in data.get("sources", []):  
            match = next((r for r in results if r["url"] == source.get("url")), None)  
            valid_sources.append(match or source)  
        return {**data, "sources": valid_sources}  
  
class ReportSynthesizer:  
    """Compiles final answers with citations"""  
      
    @staticmethod  
    def generate_report(question: str, facts: List[Dict], memory: AgentMemory) -> str:  
        """  
        Create final answer document  
          
        Args:  
            question: Original research question  
            facts: List of extracted facts  
            memory: Current agent memory  
              
        Returns:  
            Formatted answer with citations  
        """  
        facts_str = "\n".join(  
            f"Fact {i}: {f['answer']} (Confidence: {f['confidence']})"  
            for i, f in enumerate(facts, 1)  
        )  
          
        return LLMHandler.generate_response([  
            {"role": "system", "content": f"""Compile answer using:  
{memory.get_context()}  
  
Include [number] citations name and sources URL list as Sources and include confidence score for the answer."""},  
            {"role": "user", "content": f"Question: {question}\nFacts:\n{facts_str}"}  
        ], temperature=0.2)  
        
  
# =================  
# 6. Orchestrator  
# =================  
class ResearchAgent:  
    """Main controller for research operations"""  
      
    def __init__(self, memory: AgentMemory = None):  
        self.memory = memory or AgentMemory()  
          
    def execute_query(self, question: str, debug: bool = True) -> str:  
        """  
        Full research pipeline execution  
          
        Args:  
            question: Research question to investigate  
            debug: Enable debugging output  
              
        Returns:  
            Final formatted answer  
        """  
        self.memory.add_interaction(question, "")  
          
        try:  
            # Planning phase  
            plan = ResearchPlanner.create_plan(question, self.memory)  
            if debug:  
                print("Research Plan:", json.dumps(plan, indent=2))  
              
            # Execution phase  
            facts = []  
            for step in plan.get("steps", []):  
                if step["action"] == "web_search":  
                    results = WebSearch.execute(step["query"])  
                    self.memory.add_search(step["query"], results)  
                      
                    extracted = FactExtractor.process_results(question, results)  
                    facts.append(extracted)  
                      
                    if debug:  
                        print(f"Search: {step['query']} -> {len(results)} results")  
              
            # Synthesis phase  
            answer = ReportSynthesizer.generate_report(question, facts, self.memory)  
            self.memory.add_interaction(question, answer)  
              
            return answer  
          
        except Exception as e:  
            self.memory.add_interaction(question, f"Error: {str(e)}")  
            return f"Research failed: {str(e)}"  
  
# ===========================
# PoC Example Usage  
# ===========================
if __name__ == "__main__":  
    agent = ResearchAgent()  
    questions = [  
        "What was Tesla's stock price when Model S launched?",  
        "How does that compare to current price?",  
        "What were the main factors in the price change?"  
    ]  
      
    for question in questions:  
        print(f"\n[QUESTION] {question}")  
        start = time.time()  
          
        answer = agent.execute_query(question, debug=True)  
          
        print(f"\n[ANSWER] ({time.time()-start:.1f}s)")  
        print(answer)  
        print("\n" + "="*50)

