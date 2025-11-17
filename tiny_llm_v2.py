import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# 1. LOAD DATA -------------------------------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Dataset length: {len(text)} chars, vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s: str):
    return [stoi[c] for c in s if c in stoi]

def decode(indices):
    return "".join(itos[i] for i in indices)

data = torch.tensor(encode(text), dtype=torch.long)
if data.numel() < 20:
    raise ValueError("data.txt is too small, add more lines :)")

# 2. BUILD SEQ DATA (context of N chars) -----------------------
block_size = 8  # how many previous chars the model can see
inputs = []
targets = []

for i in range(len(data) - block_size):
    # context: block_size chars
    chunk = data[i : i + block_size]
    # target: the next char
    target = data[i + block_size]
    inputs.append(chunk)
    targets.append(target)

inputs = torch.stack(inputs)   # (N, block_size)
targets = torch.stack(targets) # (N,)
num_samples = inputs.size(0)
print(f"Number of (context, target) pairs: {num_samples}")

# simple train/val split
split = int(0.9 * num_samples)
x_train, x_val = inputs[:split], inputs[split:]
y_train, y_val = targets[:split], targets[split:]

# 3. TRAINING CONFIG -------------------------------------------
batch_size = 64
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
n_hidden = 128
n_embd = 32  # size of embedding per character

def get_batch(split_name: str):
    if split_name == "train":
        X, Y = x_train, y_train
    else:
        X, Y = x_val, y_val

    if X.size(0) < batch_size:
        # if dataset is tiny, just repeat samples
        idx = torch.randint(0, X.size(0), (batch_size,))
    else:
        idx = torch.randint(0, X.size(0), (batch_size,))
    xb = X[idx].to(device)
    yb = Y[idx].to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss(model, num_batches: int = 50):
    model.eval()
    losses = {"train": [], "val": []}
    for split_name in ["train", "val"]:
        for _ in range(num_batches):
            xb, yb = get_batch(split_name)
            _, loss = model(xb, yb)
            losses[split_name].append(loss.item())
    model.train()
    return {k: sum(v)/len(v) for k, v in losses.items()}

# 4. MODEL: small MLP with context -----------------------------

class TinyContextModel(nn.Module):
    """
    Slightly smarter tiny LLM:
    - Looks at last `block_size` chars
    - Uses embedding + small MLP
    """

    def __init__(self, vocab_size, n_embd, n_hidden, block_size):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, n_embd)  # each char -> vector
        self.net = nn.Sequential(
            nn.Linear(block_size * n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, vocab_size),
        )

    def forward(self, idx, targets=None):
        # idx: (B, T=block_size)
        B, T = idx.shape
        # embed -> (B, T, n_embd)
        x = self.embed(idx)
        # flatten context -> (B, T * n_embd)
        x = x.view(B, T * n_embd)
        logits = self.net(x)  # (B, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, context_ids: torch.Tensor, max_new_tokens: int = 100):
        """
        context_ids: 1D tensor of ids, will look at last `block_size`
        and generate more chars.
        """
        generated = context_ids.clone().to(device)
        if generated.numel() == 0:
            # if no context, start with random char
            start_id = torch.randint(0, vocab_size, (1,), device=device)
            generated = start_id

        for _ in range(max_new_tokens):
            # take last block_size chars as input
            context = generated[-block_size:]
            if context.numel() < block_size:
                # pad left with first char if not enough history
                pad = context[0].repeat(block_size - context.numel())
                context = torch.cat([pad, context], dim=0)

            context = context.unsqueeze(0)  # (1, block_size)
            logits, _ = self(context)
            probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1)[0, 0]
            generated = torch.cat([generated, next_id.unsqueeze(0)], dim=0)
        return generated.cpu()

# 5. TRAIN THE MODEL -------------------------------------------
model = TinyContextModel(vocab_size, n_embd, n_hidden, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training (context-aware model)...")
for step in range(max_iters + 1):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters:
        losses = estimate_loss(model, num_batches=30)
        print(
            f"step {step}/{max_iters}, "
            f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

print("Training finished!\n")

# quick random sample
start_ids = torch.randint(0, vocab_size, (block_size,))
sample_ids = model.generate(start_ids, max_new_tokens=200)
print("---- sample text (v2) ----")
print(decode(sample_ids.tolist()))
print("--------------------------")

