"""
ai_research_report.py

A minimal CrewAI crew that researches 2025 AI‑research trends and writes
a three‑paragraph tech‑blog summary.

Prerequisites
-------------
pip install 'crewai[tools]'

Environment
-----------
export OPENAI_API_KEY='sk‑…'
"""

from crewai import Agent, Task, Crew, Process

# ────────────────────────────────  Agents  ────────────────────────────────
researcher = Agent(
    role="Researcher",
    goal="Gather facts and insights on the topic",
    backstory="An expert analyst with access to vast knowledge.",
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Compose a concise report based on provided research",
    backstory="A talented writer who explains technical concepts clearly.",
    verbose=True,
)

# ────────────────────────────────  Tasks  ────────────────────────────────
research_task = Task(
    description=(
        "Research the latest trends in AI research for the year {year} "
        "and distill the findings into crisp bullet points."
    ),
    expected_output="A markdown list of factual bullet points on current AI research trends.",
    agent=researcher,
)

write_task = Task(
    description=(
        "Using the research data, craft a three‑paragraph summary suitable "
        "for a technical blog. The tone must remain accessible yet precise."
    ),
    expected_output="Exactly three well‑structured paragraphs of prose.",
    agent=writer,
    context=[research_task],  # pulls in the bullet‑point output
)

# ────────────────────────────────  Crew  ────────────────────────────────
ai_research_crew = Crew(
    name="AI Research Report Crew",
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # step‑by‑step execution
    verbose=True,
)

# ────────────────────────────────  Entry‑point  ────────────────────────────────
if __name__ == "__main__":
    crew_output = ai_research_crew.kickoff(inputs={"year": 2025})

    print("\n=== Final Blog Post ===\n")
    # CrewOutput.__str__ already returns the final text
    print(str(crew_output))
