#!/usr/bin/env python3
"""
coordinator_agent.py  — guaranteed image‐URL forwarding (UPDATED)

Orchestrates three downstream A2A worker agents (summarizer, QnA, image-gen)
by leveraging OpenAI’s function‐calling capabilities.  Ensures that any image
URL produced by the image‐generation agent is always appended in Markdown
format to the final assistant reply, so clients can reliably detect and
download it.
"""

import os
import uuid
import json
import requests
import openai
from flask import Flask, request, jsonify

# Import A2A request/response and data‐model classes from the shared common types
from common.types import (
    A2ARequest,
    SendTaskRequest,
    SendTaskResponse,
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart,
    TaskSendParams,
)

# ───────────────────── OpenAI Configuration ──────────────────────
openai.api_key  = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_TIMEOUT = 40
# When calling the image‐generation worker, allow up to 5 minutes
IMAGE_TIMEOUT   = int(os.getenv("IMAGE_TIMEOUT", 300))

# ────────────────── Downstream Worker Endpoints ──────────────────
SUMMARIZER_URL = "http://localhost:5001/tasks"
QNA_URL        = "http://localhost:5002/tasks"
IMAGE_URL      = "http://localhost:5003/tasks"

# ─────────────── Function Schema for Planner LLM ───────────────
# These "tools" definitions tell OpenAI which functions it may call,
# including argument schemas for validation.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_summarizer",
            "description": "Summarize a long context.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_qna",
            "description": "Answer a question given a summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary":  {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["summary", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_image_gen",
            "description": "Generate an illustrative image for a prompt.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        },
    },
]

# Create the Flask application
app = Flask("CoordinatorAgent")

# ══════════════════════ /tasks Endpoint ══════════════════════
@app.post("/tasks")
def receive_task():
    """
    Entry point for A2A JSON-RPC calls to tasks/send.
    Validates the RPC envelope, then uses an OpenAI-driven loop
    to call out to summarizer, QnA, and optionally image-gen tools.
    """
    req_json = request.get_json(force=True)

    # Parse the incoming JSON-RPC request into a typed A2ARequest
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        return _rpc_err(req_json.get("id"), f"Bad JSON-RPC: {exc}")

    # Only support tasks/send (wrapped as SendTaskRequest in A2ARequest)
    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_err(rpc_req.id, "Only tasks/send supported")

    params = rpc_req.params
    parts  = params.message.parts or []

    # Must have at least [context, question]
    if len(parts) < 2:
        return _task_fail(
            rpc_req.id,
            params.id,
            "Need two parts in the message: [context, question]"
        )

    # Extract text values
    context, question = parts[0].text, parts[1].text

    # Build the initial conversation for the planner LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are an orchestration agent. Use the tools to:\n"
                "1) summarize the context,\n"
                "2) answer the question, and\n"
                "3) optionally generate an illustrative image.\n"
                "When finished, reply in plain text (Markdown is allowed)."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]

    image_result = None  # Will hold the final image URL if generated

    # ───────────── chat / function‐call Loop ─────────────
    while True:
        # Ask OpenAI for next step, allowing it to request tool calls
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
        )
        choice = resp.choices[0]

        # If the model requests functions, execute them
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)  # record the function call request

            # Loop through each requested tool call
            for call in choice.message.tool_calls:
                fn_name = call.function.name
                args    = json.loads(call.function.arguments)

                if fn_name == "call_summarizer":
                    # Delegate to the Summarizer agent
                    result = _delegate(SUMMARIZER_URL, [args["text"]])

                elif fn_name == "call_qna":
                    # Delegate to the QnA agent, passing summary + question
                    result = _delegate(
                        QNA_URL, [args["summary"], args["question"]]
                    )

                elif fn_name == "call_image_gen":
                    # Delegate to the Image‐Generation worker
                    image_result = _delegate(IMAGE_URL, [args["prompt"]])
                    result = image_result

                else:
                    # Unknown tool name—report back
                    result = f"(unknown tool {fn_name})"

                # Append the tool’s raw output so the LLM can see it
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

            # Loop again so the assistant can continue reasoning
            continue

        # ───────── Final assistant reply ─────────
        final_text = choice.message.content.strip()

        # If an image URL was produced but not embedded, append Markdown
        if image_result and f"({image_result})" not in final_text:
            final_text += f"\n\n![generated image]({image_result})"

        # Build the completed Task result
        done_task = Task(
            id=params.id,
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text=final_text)]
                ),
            ),
        )
        # Return JSON‐RPC response with the Task
        return jsonify(
            SendTaskResponse(id=rpc_req.id, result=done_task).model_dump()
        )


# ══════════════════════ Helper Functions ══════════════════════
def _delegate(url: str, texts: list[str]) -> str:
    """
    Send a sub‐task to a downstream worker via A2A.
    Constructs a TaskSendParams from the given texts (each as a part),
    POSTs to the worker’s /tasks endpoint, and returns the first text part
    of the worker’s response message.
    """
    sub = TaskSendParams(
        id=uuid.uuid4().hex,
        message=Message(
            role="user",
            parts=[TextPart(text=t) for t in texts],
        ),
    )
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tasks/send",
        "params":  sub.model_dump(),
    }

    # Use a longer timeout for images, shorter for text tasks
    timeout = IMAGE_TIMEOUT if url == IMAGE_URL else DEFAULT_TIMEOUT
    resp    = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()

    # Parse the worker’s A2A Task result into our Task model
    task = Task.model_validate(resp.json()["result"])
    return task.status.message.parts[0].text


def _rpc_err(rpc_id, msg):
    """Return a JSON-RPC error (invalid request or method)."""
    return jsonify({
        "jsonrpc": "2.0",
        "id":      rpc_id,
        "error":   {"code": -32600, "message": msg},
    })


def _task_fail(rpc_id, task_id, msg):
    """Return a completed Task with FAILED state and an error message."""
    failed = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(role="agent", parts=[TextPart(text=msg)]),
        ),
    )
    return jsonify(
        SendTaskResponse(id=rpc_id, result=failed).model_dump()
    )


# ────────────────────────── Entry Point ──────────────────────────
if __name__ == "__main__":
    # For production, replace Flask dev server with Gunicorn/Uvicorn, etc.
    app.run(port=5000)
