import torch
import torch.nn as nn
import torch.nn.functional as F

# Use CPU or GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# 1. LOAD DATA -------------------------------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Dataset length: {len(text)} chars, vocab size: {vocab_size}")

# maps from char -> id and id -> char
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str):
    """Turn string into list of character ids."""
    return [stoi[c] for c in s if c in stoi]

def decode(indices):
    """Turn list of ids back into string."""
    return "".join(itos[i] for i in indices)

data = torch.tensor(encode(text), dtype=torch.long)
if data.numel() < 2:
    raise ValueError("data.txt is too small, add more text :)")

# training pairs: each char predicts the next char
x_all = data[:-1]
y_all = data[1:]
num_samples = x_all.size(0)

# 2. TRAINING CONFIG -------------------------------------------
batch_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 1e-2

def get_batch():
    """Pick random (input, target) character pairs."""
    idx = torch.randint(0, num_samples, (batch_size,))
    xb = x_all[idx].to(device)
    yb = y_all[idx].to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss(model, num_batches: int = 50):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb = get_batch()
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# 3. MODEL ------------------------------------------------------

class BigramLanguageModel(nn.Module):
    """Very tiny LLM: next char depends only on current char."""

    def __init__(self, vocab_size: int):
        super().__init__()
        # each token id maps directly to logits for next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B,)
        logits = self.token_embedding_table(idx)  # (B, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, context_ids: torch.Tensor, max_new_tokens: int = 100):
        """
        Given some starting ids (prompt), keep sampling next characters.
        context_ids: 1D tensor of shape (T,)
        """
        generated = context_ids.clone().to(device)
        for _ in range(max_new_tokens):
            last_id = generated[-1].unsqueeze(0)  # (1,)
            logits, _ = self(last_id)            # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)    # (1, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, next_id[0]], dim=0)  # (T+1,)
        return generated.cpu()

# 4. TRAIN THE MODEL -------------------------------------------
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
for step in range(max_iters + 1):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters:
        avg_loss = estimate_loss(model, num_batches=50)
        print(f"step {step}/{max_iters}, loss {avg_loss:.4f}")

print("Training finished!\n")

# quick random sample
start_id = torch.randint(0, vocab_size, (1,))
sample_ids = model.generate(start_id, max_new_tokens=200)
print("---- sample text ----")
print(decode(sample_ids.tolist()))
print("---------------------")

# 5. SIMPLE CHAT LOOP ------------------------------------------

def chat():
    print("\nTiny LLM chat. Type 'quit' to stop.\n")
    while True:
        user = input("You: ")
        if user.strip().lower() in ("quit", "exit", "q"):
            print("Bot: bye bye 👋")
            break

        # prompt style like in data.txt
        prompt = f"User: {user}\nBot:"
        context_ids = torch.tensor(encode(prompt), dtype=torch.long)

        if context_ids.numel() == 0:
            print("Bot: (I don't know these characters yet 😅)")
            continue

        out_ids = model.generate(context_ids, max_new_tokens=120)
        full_text = decode(out_ids.tolist())

        # cut off the prompt to show only model continuation
        reply = full_text[len(prompt):].strip()
        reply = reply.replace("\n", " ")

        if not reply:
            reply = "(tiny brain is confused 😵)"
        print("Bot:", reply)

if __name__ == "__main__":
    chat()

