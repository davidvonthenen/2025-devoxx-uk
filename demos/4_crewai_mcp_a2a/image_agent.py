# image_agent.py
"""
A CrewAI-based A2A agent that receives a single-part prompt and
returns an image URL generated with OpenAI's `dall-e-2` or `gpt-image-1` model.
Runs on http://localhost:5003/tasks
"""

import os
import uuid
import openai
from flask import Flask, request, jsonify

# Common JSON-RPC types shared by all A2A agents
from common.types import (
    A2ARequest,
    SendTaskRequest,
    SendTaskResponse,
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart
)

# CrewAI scaffolding: defines an “agent” that can call BaseTools
from crewai import Agent as CrewAgent, Task as CrewTask, Crew
from crewai.tools import BaseTool

# -------- OpenAI setup ----------
# The API key must be set in your environment before starting this agent
openai.api_key = os.environ["OPENAI_API_KEY"]

# Choose between “dall-e-2” or “gpt-image-1” via this constant
IMAGE_MODEL = "dall-e-2"  # switch to "gpt-image-1" if preferred

# Name of the LLM that powers CrewAI’s internal reasoning
LLM_NAME = os.getenv("IMAGE_LLM", "gpt-4.1-mini")
# ---------------------------------

# ---------- CrewAI Tool Definition ----------
class ImageGenTool(BaseTool):
    """
    Wraps the OpenAI image-generation API as a CrewAI tool.
    This is where you could add:
      - request_timeout
      - retry logic
      - prompt preprocessing or validation
    """
    name: str = "generate_image"
    description: str = (
        "Generate an illustrative image from a prompt and return the image URL."
    )

    def _run(self, prompt: str) -> str:
        # Submit a single 1024×1024 render request to OpenAI Images
        resp = openai.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024",
            # request_timeout=int(os.getenv("OPENAI_IMAGE_TIMEOUT", 120)),  # optional
        )

        # Always return the first URL; CrewAI will attach it to its response
        return resp.data[0].url

# Instantiate the tool and wrap it in a CrewAI agent
img_tool = ImageGenTool()

image_creator = CrewAgent(
    role="Image Creator",
    goal="Produce vivid illustrative images",
    backstory="An expert generative-artist AI.",
    tools=[img_tool],
    llm=LLM_NAME
)

# Create the Flask app to expose the /tasks JSON-RPC endpoint
app = Flask("ImageGenAgent")

# ---------------- /tasks endpoint ----------------
@app.post("/tasks")
def handle_task():
    """
    Main entrypoint for A2A requests.
    1. Parse JSON-RPC → SendTaskRequest
    2. Extract the single text part as 'prompt'
    3. Run CrewAI to generate an image URL
    4. Return a SendTaskResponse containing the URL
    """
    req_json = request.get_json(force=True)

    # Validate JSON-RPC envelope
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        return _rpc_err(req_json.get("id"), f"Bad JSON-RPC: {exc}")

    # We only support tasks/send calls here
    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_err(rpc_req.id, "Only tasks/send supported")

    params = rpc_req.params
    # Grab the first text part; fallback to a placeholder if missing
    prompt = (
        params.message.parts[0].text
        if params.message.parts
        else "(empty prompt)"
    )

    # Build a one-off CrewTask for this single prompt
    task_obj = CrewTask(
        description=f"Generate an image for the prompt: '{prompt}'",
        expected_output="Image URL",
        agent=image_creator
    )
    crew = Crew(agents=[image_creator], tasks=[task_obj], verbose=False)

    # Execute the CrewAI agent
    try:
        crew_output = crew.kickoff()
        # CrewOutput.raw holds the direct tool return if available
        image_url = (
            crew_output.raw
            if hasattr(crew_output, "raw")
            else str(crew_output)
        )
        print(f"[image_agent] Generated URL → {image_url}")
    except Exception as exc:
        # If CrewAI internals fail, surface as a task failure
        return _task_fail(rpc_req.id, params.id, f"Crew error: {exc}")

    # Wrap the URL in our Task/TaskStatus/Message hierarchy
    done_task = Task(
        id=params.id,
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(
                role="agent",
                parts=[TextPart(text=image_url)]
            )
        )
    )

    return jsonify(
        SendTaskResponse(id=rpc_req.id, result=done_task).model_dump()
    )

# ---------------- helper responses ----------------
def _rpc_err(rpc_id, msg):
    """Return a JSON-RPC <error> response."""
    return jsonify({
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32600, "message": msg}
    })


def _task_fail(rpc_id, task_id, msg):
    """Return a TaskStatus.FAILED payload when something goes wrong."""
    failed = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(
                role="agent",
                parts=[TextPart(text=msg)]
            )
        )
    )
    return jsonify(
        SendTaskResponse(id=rpc_id, result=failed).model_dump()
    )

# ---------------- run server -----------------------------
if __name__ == "__main__":
    # In production you’d swap Flask’s dev server for Gunicorn/Uvicorn/etc.
    app.run(port=5003)
