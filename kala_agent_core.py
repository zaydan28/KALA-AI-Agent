import sqlite3
import httpx
import json
from typing import List, Dict, Optional

class KalaMemoryStorage:
    """
    Manages long-term autonomous agent memory using SQLite FTS5.
    Provides ultra-fast text retrieval without the overhead of external Vector DBs.
    """
    def __init__(self, db_path: str = "kala_production_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._initialize_fts5_table()

    def _initialize_fts5_table(self):
        with self.conn:
            # Using FTS5 virtual table for optimized semantic-like keyword matching
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS agent_memory 
                USING fts5(timestamp, role, content)
            """)

    def save_interaction(self, role: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO agent_memory (timestamp, role, content) VALUES (datetime('now'), ?, ?)",
                (role, content)
            )

    def search_context(self, query: str, limit: int = 5) -> str:
        # FTS5 MATCH operator ranks results natively based on relevance (BM25 algorithm)
        cursor = self.conn.execute(
            "SELECT role, content FROM agent_memory WHERE agent_memory MATCH ? ORDER BY rank LIMIT ?",
            (query, limit)
        )
        results = cursor.fetchall()
        if not results:
            return "No relevant past context found in local memory."
        
        return "\n".join([f"{row[0].capitalize()}: {row[1]}" for row in results])


class KalaAutonomousAgent:
    """
    Core Agentic Loop integrating Nous Hermes / DeepSeek via OpenRouter API
    with self-healing memory context generation.
    """
    def __init__(self, api_key: str, model_id: str = "nousresearch/nous-hermes-3-llama-3.1-405b"):
        self.api_key = api_key
        self.model = model_id
        self.memory = KalaMemoryStorage()
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _build_contextual_prompt(self, user_input: str, context: str) -> List[Dict[str, str]]:
        """Assembles the prompt tree with grounding guardrails."""
        return [
            {
                "role": "system", 
                "content": (
                    "You are KALA, an autonomous infrastructure guardian running on an Ubuntu VPS. "
                    "Your primary function is to monitor, orchestrate, and secure production environments. "
                    "Base your responses strictly on the provided context."
                )
            },
            {"role": "system", "content": f"SYSTEM MEMORY (FTS5 Context):\n{context}"},
            {"role": "user", "content": user_input}
        ]

    def process_task(self, user_input: str) -> Optional[str]:
        """
        The main autonomous loop: 
        Retrieve Memory -> Assemble Prompt -> Execute LLM -> Store State.
        """
        # 1. Retrieve relevant memory state
        context = self.memory.search_context(user_input)
        
        # 2. Construct robust payloads
        messages = self._build_contextual_prompt(user_input, context)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/zaydan28/KALA-Agent" 
        }
        
        # Setting low temperature for deterministic infrastructure outputs
        payload = {"model": self.model, "messages": messages, "temperature": 0.1}

        # 3. Execute LLM Call safely using httpx
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                
                ai_response = response.json()["choices"][0]["message"]["content"]
                
                # 4. Save state back to memory for continuous learning
                self.memory.save_interaction("user", user_input)
                self.memory.save_interaction("assistant", ai_response)
                
                return ai_response
                
        except httpx.HTTPError as e:
            print(f"[ERROR] Agent Execution Pipeline Failed: {e}")
            return None

# ==========================================
# Production Entrypoint / Daemon Execution
# ==========================================
if __name__ == "__main__":
    # Example initialization in a production daemon environment
    import os
    API_KEY = os.getenv("OPENROUTER_API_KEY", "your_api_key_here")
    
    agent = KalaAutonomousAgent(api_key=API_KEY)
    
    # Simulating an event-driven task
    print("Initiating KALA Agent Loop...")
    task = "Analyze recent Nginx error logs on the Ubuntu VPS and suggest fixes."
    
    result = agent.process_task(task)
    print(f"Agent Output:\n{result}")
