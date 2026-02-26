"""
Query Generation Operations for ARK

This module provides the QueryGenerator class for generating augmented queries
from knowledge graph subgraphs using LLMs.
"""

import json
import os
from typing import Dict, List, Any

# Import prompts
from src.kg.prompts.query_gen import Prompt as QUERY_GEN_PROMPT


class QueryGenerator:
    """Generate augmented queries using LLMs"""

    def __init__(
        self,
        llm_provider: str = "gemini",
        llm_model: str = "gemini-2.0-flash",
        num_queries: int = 10,
        llm_config: dict = None,
    ):
        """
        Args:
            llm_provider: LLM provider (gemini, openai, etc.)
            llm_model: Specific model name
            num_queries: Number of queries to generate per subgraph
            llm_config: LLM API configuration dictionary
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.num_queries = num_queries
        self.llm_config = llm_config or {}

        # Initialize LLM client
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client based on provider"""
        if self.llm_provider == "gemini":
            try:
                import google.generativeai as genai
                # Get API key from config or environment
                api_key = self.llm_config.get('gemini', {}).get('api_key') or os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.llm_model)
                self.llm_func = self._generate_with_gemini
            except ImportError:
                raise ImportError(
                    "Please install google-generativeai: "
                    "pip install google-generativeai"
                )
        elif self.llm_provider == "openai":
            try:
                from openai import OpenAI
                # Get API key and base_url from config or environment
                api_key = self.llm_config.get('gpt', {}).get('api_key') or os.environ.get("OPENAI_API_KEY")
                base_url = self.llm_config.get('gpt', {}).get('base_url')

                if base_url:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                else:
                    self.client = OpenAI(api_key=api_key)
                self.llm_func = self._generate_with_openai
            except ImportError:
                raise ImportError(
                    "Please install openai: pip install openai"
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _generate_with_gemini(self, prompt: str) -> str:
        """Generate text using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating with Gemini: {e}")
            return None

    def _generate_with_openai(self, prompt: str) -> str:
        """Generate text using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Output ONLY valid JSON without any explanation or thinking process."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with OpenAI: {e}")
            return None

    def generate_queries_from_subgraph(
        self,
        original_question: str,
        answer: str,
        entities: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate confusing queries from a subgraph

        Args:
            original_question: The original question
            answer: The ground truth answer
            entities: List of entity dictionaries with name, type, descriptions

        Returns:
            List of generated queries
        """
        # Prepare input in the expected format
        input_data = {
            "original_question": original_question,
            "answer": answer,
            "entities": entities
        }

        # Format the prompt
        prompt = QUERY_GEN_PROMPT.format(Input=json.dumps(input_data, indent=2))

        # Generate queries
        response = self.llm_func(prompt)

        if response is None:
            return []

        try:
            # Parse JSON response
            response_text = response.strip()

            # Strip markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = '\n'.join(lines)

            # Find the first JSON object/array (starts with { or [)
            json_start = -1
            for i, char in enumerate(response_text):
                if char in ['{', '[']:
                    json_start = i
                    break

            if json_start >= 0:
                response_text = response_text[json_start:]

            # Find the last closing brace/bracket
            json_end = -1
            for i in range(len(response_text) - 1, -1, -1):
                if response_text[i] in ['}', ']']:
                    json_end = i + 1
                    break

            if json_end > 0:
                response_text = response_text[:json_end]

            result = json.loads(response_text)
            queries = result.get("confusing_questions", [])
            print(f"[DEBUG] Parsed {len(queries)} queries")
            return queries[:self.num_queries]
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response text (first 500 chars): {response_text[:500] if response_text else 'None'}")
            return []

    def process_subgraph_file(
        self,
        subgraph_path: str,
        original_data: Dict[str, Any]
    ) -> List[str]:
        """
        Process a single subgraph file and generate queries

        Args:
            subgraph_path: Path to the subgraph json file
            original_data: Original question-answer data

        Returns:
            List of generated queries
        """
        # Load subgraph (pretty-printed jsonl with indent)
        subgraph_data = []
        with open(subgraph_path, 'r', encoding='utf-8') as f:
            content = f.read()
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content):
            if content[idx] in ' \t\n\r':
                idx += 1
                continue
            obj, end = decoder.raw_decode(content, idx)
            subgraph_data.append(obj)
            idx = end

        # Convert to entities list
        entities = []
        for entity_dict in subgraph_data:
            entities.append({
                "name": entity_dict.get('entity_name', '').strip('"'),  # Remove quotes
                "type": entity_dict.get('entity_type', 'Unknown').strip('"'),
                "descriptions": [entity_dict.get('description', '')]
            })

        # Get original question and answer
        original_question = original_data.get("input", original_data.get("question", ""))
        # Handle both "answer" (string) and "answers" (list) formats
        answer = original_data.get("answer", "")
        if not answer:
            answers = original_data.get("answers", [])
            answer = answers[0] if answers else original_data.get("output", "")

        # Generate queries
        queries = self.generate_queries_from_subgraph(
            original_question=original_question,
            answer=answer,
            entities=entities
        )

        return queries
