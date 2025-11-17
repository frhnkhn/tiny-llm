from flask import Flask, request, render_template_string
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- LOAD MODEL CHECKPOINT ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

ckpt_path = "tiny_llm_v2_ckpt.pt"
print(f"Loading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location=device)

stoi = ckpt["stoi"]
itos = ckpt["itos"]
config = ckpt["config"]

vocab_size = config["vocab_size"]
block_size = config["block_size"]
n_embd = config["n_embd"]
n_hidden = config["n_hidden"]

def encode(s: str):
    return [stoi[c] for c in s if c in stoi]

def decode(indices):
    return "".join(itos[i] for i in indices)

class TinyContextModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden, block_size):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.net = nn.Sequential(
            nn.Linear(block_size * n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, vocab_size),
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)          # (B, T, n_embd)
        x = x.view(B, T * n_embd)    # (B, T*n_embd)
        logits = self.net(x)         # (B, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, context_ids: torch.Tensor, max_new_tokens: int = 100):
        generated = context_ids.clone().to(device)
        if generated.numel() == 0:
            start_id = torch.randint(0, vocab_size, (1,), device=device)
            generated = start_id

        for _ in range(max_new_tokens):
            context = generated[-block_size:]
            if context.numel() < block_size:
                pad = context[0].repeat(block_size - context.numel())
                context = torch.cat([pad, context], dim=0)

            context = context.unsqueeze(0)  # (1, block_size)
            logits, _ = self(context)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)[0, 0]
            generated = torch.cat([generated, next_id.unsqueeze(0)], dim=0)
        return generated.cpu()

# create and load model
model = TinyContextModel(vocab_size, n_embd, n_hidden, block_size).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("Model loaded.")

# --------- FLASK APP ---------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Farhan's Tiny LLM v2</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; }
        h1 { font-size: 24px; }
        .box { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 16px; }
        label { font-weight: 600; }
        input[type="text"] { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #ccc; }
        button { margin-top: 8px; padding: 8px 16px; border-radius: 6px; border: none; cursor: pointer; }
        button:hover { opacity: 0.9; }
        .user { font-weight: 600; }
        .bot { margin-top: 8px; }
    </style>
</head>
<body>
    <h1>Farhan's Tiny LLM v2 🤖</h1>
    <p>A tiny character-level model trained on my own data, running on my Mac.</p>

    <form method="POST">
        <div class="box">
            <label for="user_text">Your message:</label><br/>
            <input type="text" id="user_text" name="user_text" value="{{ user_text|default('') }}" autocomplete="off"/>
            <button type="submit">Ask</button>
        </div>
    </form>

    {% if user_text %}
    <div class="box">
        <div class="user">You: {{ user_text }}</div>
        <div class="bot">Bot: {{ bot_reply }}</div>
    </div>
    {% endif %}
</body>
</html>
"""

def generate_reply(user_text: str) -> str:
    prompt = f"User: {user_text}\nBot:"
    ids = torch.tensor(encode(prompt), dtype=torch.long)
    if ids.numel() == 0:
        return "(I don't know these characters yet 😅)"
    out_ids = model.generate(ids, max_new_tokens=120)
    full_text = decode(out_ids.tolist())
    reply = full_text[len(prompt):].strip().replace("\\n", " ")
    return reply or "(tiny brain v2 is confused 😵)"

@app.route("/", methods=["GET", "POST"])
def index():
    user_text = ""
    bot_reply = ""
    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        if user_text:
            bot_reply = generate_reply(user_text)
    return render_template_string(HTML_TEMPLATE, user_text=user_text, bot_reply=bot_reply)

if __name__ == "__main__":
    app.run(debug=True)

