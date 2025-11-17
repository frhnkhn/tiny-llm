from flask import Flask, render_template, request, jsonify
import requests

# --------- OLLAMA CONFIG ---------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # change if you pulled a different model

app = Flask(__name__)


def generate_reply(user_text: str) -> str:
    """
    Send the user's message to the local Ollama model and get a reply.
    """
    # You can tweak this prompt style to match how you want it to talk
    prompt = (
        "You are a friendly, helpful AI assistant chatting with a student named Farhan. "
        "Answer clearly and concisely.\n\n"
        f"User: {user_text}\nAssistant:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,  # get one full response as JSON
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("response", "").strip()
        if not reply:
            return "(model returned an empty response 😅)"
        return reply
    except Exception as e:
        print("Error talking to Ollama:", e)
        return "(error talking to local model 😵)"


# --------- FLASK ROUTES ---------


@app.route("/")
def index():
    # uses templates/index.html (your existing UI)
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"reply": "(say something first 😄)"})

    reply = generate_reply(user_text)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    # debug=True is fine for local dev
    app.run(debug=True)
