#!/usr/bin/env python3
"""
mcp_llama_agent.py

CrewAI demo:
  • Queries PubMed via pubmedmcp@0.1.3
  • Summarises 2025 AI literature
using a local Llama GGUF model (llama-cpp-python, no LangChain).
"""

from __future__ import annotations
import os
from typing import List, Dict, Union, Optional, Any

from crewai import Agent, Task, Crew, Process, BaseLLM
from llama_cpp import Llama              # pip install llama-cpp-python
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.crewai_adapter import CrewAIAdapter

# ───────────────────────────────────────────────────────────────
# 1. Path to your local GGUF model
# ───────────────────────────────────────────────────────────────
GGUF_PATH = "/Users/vonthd/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


# ───────────────────────────────────────────────────────────────
# 2. Minimal CrewAI Llama-cpp wrapper
# ───────────────────────────────────────────────────────────────
class LlamaCppLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        super().__init__(model=model_path)
        self.client = Llama(model_path=model_path, n_ctx=n_ctx, temperature=temperature)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = (
            messages
            if isinstance(messages, str)
            else self._chat_to_prompt(messages)
        ).strip()

        resp = self.client.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp["choices"][0]["text"].lstrip()

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096

    @staticmethod
    def _chat_to_prompt(chat: List[Dict[str, str]]) -> str:
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in chat]
        lines.append("Assistant:")
        return "\n".join(lines)


# Instantiate once and reuse
local_llm = LlamaCppLLM(GGUF_PATH)


def _build_crew(pubmed_tools: Union[List[Any], Any], year: int) -> Crew:
    """
    Build a two‐agent Crew using the PubMed tool from MCP.
    If pubmed_tools is a list, we pick the first entry.
    """
    tools_list = pubmed_tools if isinstance(pubmed_tools, list) else [pubmed_tools]
    if not tools_list:
        raise ValueError("No MCP tools available — check your StdioServerParameters.")

    # select the PubMed tool
    pubmed_tool = tools_list[0]

    researcher = Agent(
        role="Researcher",
        goal="Gather facts and insights on the topic",
        backstory="An expert analyst with privileged PubMed access.",
        llm=local_llm,
        tools=[pubmed_tool],
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer",
        goal="Compose a concise report based on provided research",
        backstory="A talented writer who explains technical concepts clearly.",
        llm=local_llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Use the PubMed tool to research the latest AI‐research trends in {year} "
            "with emphasis on deep‐learning architectures, agentic AI and safety alignment. "
            "Summarise the findings in bullet points."
        ),
        expected_output="Markdown bullet list of factual points.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Turn the research bullets into *exactly* three coherent paragraphs "
            "for a technical blog post, accessible yet precise."
        ),
        expected_output="Three paragraphs of prose.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        name="AI Research Report Crew",
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )


def main() -> None:
    server_params = StdioServerParameters(
        command="uvx",
        args=["--quiet", "pubmedmcp@0.1.3"],
        env={"UV_PYTHON": "3.11", **os.environ},
    )

    with MCPAdapt(server_params, CrewAIAdapter()) as tools:
        crew = _build_crew(tools, year=2025)
        result = crew.kickoff(inputs={"year": 2025})

    print("\n=== Final Blog Post ===\n")
    print(result)


if __name__ == "__main__":
    # Don’t forget: export OPENAI_API_KEY if any MCP tools need it.
    main()
