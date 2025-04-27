# summarizer_agent.py

"""
Summarizer Agent for the Agent2Agent (A2A) Protocol

This service listens for A2A JSON-RPC tasks on `/tasks` (POST) and
returns a 2–3 sentence summary of the provided text using OpenAI’s LLM.

It uses the shared A2A models from the `common` package to parse incoming
requests (A2ARequest / SendTaskRequest) and to construct proper JSON-RPC
responses (SendTaskResponse / Task).
"""

import os
import openai
from flask import Flask, request, jsonify
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

# ----------------------------------------------------------------------
# Configure OpenAI
# ----------------------------------------------------------------------

# Read the OpenAI API key from the environment
openai.api_key = os.environ["OPENAI_API_KEY"]
# Default model can be overridden via OPENAI_MODEL env var
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ----------------------------------------------------------------------
# Flask App Initialization
# ----------------------------------------------------------------------

app = Flask("SummarizerAgent")

# ----------------------------------------------------------------------
# A2A Task Endpoint
# ----------------------------------------------------------------------

@app.post("/tasks")
def handle_task():
    """
    Handle incoming A2A `tasks/send` requests.
    1. Parse JSON-RPC request into A2ARequest.
    2. Ensure it's a SendTaskRequest.
    3. Extract the text to summarize from the first message part.
    4. Call the summarization helper.
    5. Wrap the summary in a Task result and return via JSON-RPC.
    """
    # Retrieve the raw JSON from the HTTP request
    req_json = request.get_json(force=True)

    # 1. Validate and parse the JSON-RPC payload
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        # Return a JSON-RPC error if parsing fails
        return _rpc_err(req_json.get("id"), f"Invalid JSON-RPC: {exc}")

    # 2. Ensure this is a tasks/send call (SendTaskRequest)
    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_err(rpc_req.id, "Unsupported method; only tasks/send is handled")

    # 3. Extract text from the incoming A2A task parameters
    params = rpc_req.params
    text_to_sum = ""
    if params.message.parts:
        text_to_sum = params.message.parts[0].text

    # 4. Generate the summary via OpenAI
    summary = _summarize(text_to_sum)

    # 5. Build the Task result (completed) with the summary as the agent message
    done_task = Task(
        id=params.id,
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text=summary)])
        )
    )

    # Wrap in a JSON-RPC SendTaskResponse and return
    response = SendTaskResponse(id=rpc_req.id, result=done_task)
    return jsonify(response.model_dump())

# ----------------------------------------------------------------------
# Summarization Helper
# ----------------------------------------------------------------------

def _summarize(text: str) -> str:
    """
    Call OpenAI's Chat Completion API to summarize the input text
    in 2–3 sentences. Returns the trimmed summary string.
    """
    # Construct the user prompt
    prompt = f"Summarize the following text in 2–3 sentences:\n\n{text}"

    # Send the request to OpenAI
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    answer = resp.choices[0].message.content.strip()

    # Debug output for development (split logs clearly)
    print("=== Summarization Debug ===")
    print(f"Input: {text}")
    print(f"Summary: {answer}")
    print("=" * 40)

    return answer

# ----------------------------------------------------------------------
# JSON-RPC Error Helper
# ----------------------------------------------------------------------

def _rpc_err(rpc_id, msg):
    """
    Return a JSON-RPC 2.0 error response with code -32600 (Invalid Request).
    """
    return jsonify({
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32600, "message": msg}
    })

# ----------------------------------------------------------------------
# App Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Launch the Flask development server on port 5001
    app.run(port=5001)
