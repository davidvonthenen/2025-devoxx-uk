"""
ai_research_report_mcp.py
────────────────────────────────────────────────────────────────────────────
Runs a CrewAI team that searches PubMed through MCP and writes a blog post.
"""

from __future__ import annotations
import os

from crewai import Agent, Task, Crew, Process
from mcp import StdioServerParameters
from mcpadapt.core import MCPAdapt
from mcpadapt.crewai_adapter import CrewAIAdapter


def _build_crew(pubmed_tools, year: int) -> Crew:
    """Create agents, tasks and the crew (no kickoff yet)."""
    researcher = Agent(
        role="Researcher",
        goal="Gather facts and insights on the topic",
        backstory="An expert analyst with privileged PubMed access.",
        tools=pubmed_tools,           # live MCP tools here
        verbose=True,
    )

    writer = Agent(
        role="Writer",
        goal="Compose a concise report based on provided research",
        backstory="A talented writer who explains technical concepts clearly.",
        verbose=True,
    )

    research_task = Task(
        description=(
            f"Use the PubMed tools to research the latest AI-research trends in {year} "
            "with emphasis on novel deep-learning architectures, agentic AI and safety "
            "alignment. Summarise the findings in bullet points."
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

    # Keep the MCP server alive for the whole run
    with MCPAdapt(server_params, CrewAIAdapter()) as tools:
        crew = _build_crew(tools, year=2025)
        result = crew.kickoff(inputs={"year": 2025})

    print("\n=== Final Blog Post ===\n")
    print(result)            # MCP server is stopped *after* this line


if __name__ == "__main__":
    # Make sure your key is exported:  export OPENAI_API_KEY=sk-…
    main()
