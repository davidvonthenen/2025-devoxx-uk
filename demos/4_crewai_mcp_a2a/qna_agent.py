# qna_agent.py
# Agent2Agent Q&A Worker Agent using OpenAI

import os
import openai
from flask import Flask, request, jsonify
from common.types import (
    A2ARequest,          # Model to parse incoming A2A JSON-RPC requests
    SendTaskRequest,     # Specific request type for tasks/send
    SendTaskResponse,    # Response wrapper for tasks/send
    Task,                # A2A Task result model
    TaskStatus,          # Task status details (state, message)
    TaskState,           # Enumeration of possible task states
    Message,             # Message container (role + parts)
    TextPart             # Text part for messages
)

# Retrieve OpenAI credentials and model name from environment
openai.api_key = os.environ["OPENAI_API_KEY"]
# Default to GPT-4 variant; ensure this matches your account’s availability
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Initialize Flask app for this agent
app = Flask("QnAAgent")

@app.post("/tasks")
def handle_task():
    """
    Entry point for A2A 'tasks/send' requests.
    Parses the JSON-RPC payload into a SendTaskRequest, extracts context and question,
    calls OpenAI to answer, and returns a TaskResult.
    """
    req_json = request.get_json(force=True)
    try:
        # Validate and parse incoming A2A request
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        # Malformed JSON-RPC input
        return _rpc_err(req_json.get("id"), f"Invalid JSON-RPC: {exc}")

    # Ensure this is a tasks/send call
    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_err(rpc_req.id, "Unsupported method")

    # Extract parameters from the parsed request
    params = rpc_req.params
    parts = params.message.parts or []
    # We require at least two parts: [context, question]
    if len(parts) < 2:
        return _task_fail(rpc_req.id, params.id, "Need context + question")

    # Assign context and question text
    context, question = parts[0].text, parts[1].text
    # Call OpenAI to get the answer, ensuring a non-empty string return
    answer = _answer(context, question) or "(no answer returned)"

    # Build the completed Task with the answer embedded as an agent message
    done_task = Task(
        id=params.id,
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text=answer)])
        )
    )
    # Wrap in a JSON-RPC SendTaskResponse and serialize
    return jsonify(SendTaskResponse(id=rpc_req.id, result=done_task).model_dump())

# ----------------------- Helper Functions -----------------------

def _answer(context: str, question: str) -> str:
    """
    Sends the context and question to OpenAI's chat completion endpoint.
    Returns the assistant’s reply, or an empty string on error.
    """
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. Answer strictly using the provided context."},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": question}
    ]
    try:
        # Use the new v5 openai.chat.completions interface
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=256
        )
        # Safely extract the assistant’s content from the response
        choice = resp.choices[0] if resp.choices else None
        text = choice.message.content if choice and choice.message.content else ""
        answer = text.strip()

        # Debug: log question and answer to console
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 40)

        return answer
    except Exception:
        # On API or parsing errors, return empty string instead of None
        return ""


def _task_fail(rpc_id, task_id, msg):
    """
    Helper to return a TaskResult marked as FAILED for a given task.
    """
    failed = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(role="agent", parts=[TextPart(text=msg)])
        )
    )
    return jsonify(SendTaskResponse(id=rpc_id, result=failed).model_dump())


def _rpc_err(rpc_id, msg):
    """
    Helper to construct a JSON-RPC error response for malformed RPC calls.
    """
    return jsonify({
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32600, "message": msg}
    })

if __name__ == "__main__":
    # Launch the agent on port 5002
    app.run(port=5002)
