# client.py ── saves images in the current directory and keeps full SAS query-strings
"""
CLI client for the Coordinator agent — downloads any image URL (with its
signed query string) and prints the agent’s reply.

Usage:
    python client.py  "context here"  "question here"

If no args are given, a default Artemis example is used.
"""

import sys
import uuid
import re
import json
import datetime
from pathlib import Path
import requests

# Import the shared A2A message and task parameter models
from common.types import Message, TextPart, TaskSendParams

# ──────────────────────────────────────────────────────────────────────────────
COORD_URL  = "http://localhost:5000/tasks"  # Coordinator Agent A2A endpoint
JSONRPC_ID = 1                              # Fixed JSON-RPC request ID

# Regex patterns to locate image URLs in responses
MD_IMG_REGEX = re.compile(r"!\[.*?\]\((.*?)\)")  # Markdown-style ![alt](url)
URL_REGEX    = re.compile(r"https?://\S+", re.I)  # Any http(s) URL

DOWNLOAD_DIR = Path(".")  # Directory where downloaded images will be saved
# ──────────────────────────────────────────────────────────────────────────────


def build_payload(context: str, question: str) -> dict:
    """
    Construct the JSON-RPC payload for the Coordinator's /tasks/send call.
    - Wraps the context and question as two TextPart entries in a Message.
    - Generates a unique TaskSendParams.id to track the task.
    """
    msg = Message(
        role="user",
        parts=[
            TextPart(text=context),
            TextPart(text=question),
        ],
    )
    params = TaskSendParams(id=uuid.uuid4().hex, message=msg)
    return {
        "jsonrpc": "2.0",
        "id": JSONRPC_ID,
        "method": "tasks/send",
        "params": params.model_dump(),  # Convert Pydantic model to dict
    }


def safe_request(url: str, payload: dict) -> dict:
    """
    Send a POST request with JSON payload to the given URL.
    Raises an exception for non-2xx responses.
    Returns the parsed JSON response.
    """
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def extract_parts(result_json: dict) -> list[dict]:
    """
    Navigate the A2A TaskResult JSON to extract the list of
    message parts under result.status.message.parts.
    """
    try:
        return result_json["result"]["status"]["message"]["parts"]
    except KeyError:
        # If the structure is unexpected, log the full JSON for debugging
        print("[WARN] Unexpected JSON structure:")
        print(json.dumps(result_json, indent=2))
        return []


# ───────────────────────── image helpers ──────────────────────────
def extract_image_url(text: str) -> str | None:
    """
    Search the text for an image URL.
    Prefer Markdown-style links first; fall back to any raw URL.
    """
    md_match = MD_IMG_REGEX.search(text)
    if md_match:
        return md_match.group(1)
    raw_match = URL_REGEX.search(text)
    return raw_match.group(0) if raw_match else None


def download_image(url: str) -> Path | None:
    """
    Download the image at `url` into DOWNLOAD_DIR.
    Returns the Path to the saved file, or None on failure.
    """
    # Extract file extension (strip any query string)
    ext = url.split(".")[-1].split("?")[0]
    # Timestamped filename to avoid collisions
    fname = f"IMG_{datetime.datetime.now():%Y%m%d_%H%M%S}.{ext}"
    dest = DOWNLOAD_DIR / fname

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as exc:
        print(f"[IMAGE-DL] Failed to download {url}: {exc}")
        return None


# ───────────────────────── main workflow ──────────────────────────
def send(context: str, question: str):
    """
    1. Build the JSON-RPC payload for the Coordinator.
    2. Send it and parse the response.
    3. Extract text parts and look for an image URL.
    4. If found, download the image; then display the full reply text.
    """
    payload = build_payload(context, question)
    data    = safe_request(COORD_URL, payload)

    # Handle any JSON-RPC error returned by the Coordinator
    if data.get("error"):
        print("Coordinator error:", json.dumps(data["error"], indent=2))
        return

    # Get the list of text/image parts from the TaskResult
    parts = extract_parts(data)
    if not parts:
        print("[No reply parts]")
        return

    # Reassemble the agent's reply text
    answer_text = "\n".join(p.get("text", "") for p in parts)

    # Attempt to download any discovered image URL
    img_url = extract_image_url(answer_text)
    if img_url:
        path = download_image(img_url)
        if path:
            print(f"[Image saved → {path.name}]")
        else:
            print("[Image download failed]")
    else:
        print("[No image URL found]")

    # Finally, print the textual reply
    print("\n----- AGENT REPLY -----\n")
    print(answer_text)


if __name__ == "__main__":
    # Allow passing custom context & question via CLI arguments
    if len(sys.argv) >= 3:
        ctx, q = sys.argv[1], sys.argv[2]
    else:
        # Default example: Lunar Artemis engineering context + mixed media request
        ctx = (
            """From the moment I first read the mission statement—“NASA's Artemis program aims to establish a sustainable human presence on the Moon”—I felt like a kid again, staring up at the night sky and daring to dream. Back then, my only lunar ambition was convincing my parents I really needed that telescope for my birthday. Today, I'm one of the engineers wrestling with that very statement, unpacking every word to turn it into reality. My coffee-fueled mornings begin with whiteboards scrawled in orbital mechanics and habitat schematics, each doodle a stubborn reminder that sustaining life on the Moon is more than planting a flag and taking selfies.

            Transporting large payloads isn't as cinematic as in the movies—there are no dramatic launch sequences with glowing thrusters that hover above a lunar cliff. Instead, there's a delicate ballet of mass budgets, structural stiffness, and propulsion trade-offs. I remember when our team tested a prototype cargo lander's legs in the desert; watching that machine wobble under load was like watching a toddler try to stand after one too many birthday cakes. We redesigned the leg geometry three times because telescoping struts that seemed rock-solid on paper kept bending like wet spaghetti under real-world stresses.

            Then comes the marathon of long-term life-support. It's one thing to keep a plant alive in a lab for a week, quite another to ensure a greenhouse under lunar regolith thrives for months. I've spent countless hours analyzing water-recycling loops and atmospheric scrubbers, peering at sensor readouts that hint at biological processes more temperamental than my houseplants back on Earth. Every hiccup—a droplet of condensate where it shouldn't be, a CO₂ spike at 3 a.m.—sparks spirited debates over algorithm tweaks and hardware tweaks. Truth be told, there's a certain thrill in chasing down these microscopic gremlins; it's like gardening in zero gravity.

            Shielding astronauts from cosmic radiation might sound like wrapping them in lead blankets, but the real solution is far more elegant—and far more complex. We're experimenting with regolith-packed walls, magnetic deflection fields, and even polyethylene composites that absorb high-energy particles. I still chuckle at my first simulation, where a single solar flare lobbed enough radiation at our habitat model to trigger alarms louder than a 1980s arcade game. Those early failures taught us humility: Mother Nature will always find a way to remind you that you're not the smartest one in the room.

            Looking back, designing for Artemis has been the most exhilarating engineering challenge of my career. Every setback has sharpened our ingenuity, every late-night breakthrough reaffirmed why we chose this path. And here's my call to action: whether you're a student sketching rockets on a napkin or a seasoned researcher tweaking life-support membranes, lean in. The Moon is waiting not for spectators, but for problem-solvers ready to turn “what if” into “what next.” Let's make Artemis the beginning of humanity's greatest adventure yet."""
        )
        q = (
            "Create an illustration capturing this context, and what is one major engineering challenge not mentioned above?"
        )
    send(ctx, q)
