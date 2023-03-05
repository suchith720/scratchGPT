import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------
data_file = "/home/scai/phd/aiz218323/scratch/Courses/nanoGPT/input.txt"



with open(data_file, 'r', encoding="utf-8") as file:
    content = file.read()

print(f"Number of characters : {len(content)}")

characters = sorted(list(set(content)))
vocab_size = len(characters)
print(f"Vocabulary size : {len(characters)}")

stoi = { c:i for i, c in enumerate(characters)}
itos = { i:c for i, c in enumerate(characters)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda x: "".join([itos[i] for i in x])


# Train-test split
data = torch.tensor(encode(content), dtype=torch.long)
n_train = int(0.9*len(data))
train_data = data[:n_train]
valid_data = data[n_train:]

def get_batch(split):
    data = train_data if split == "train" else valid_data
    idxs = torch.randint(len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[idx:idx+block_size] for idx in idxs])
    y = torch.stack([data[idx+1:idx+block_size+1] for idx in idxs])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.token_embedding_table(x) # (B,T,C)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

model = BigramLanguageModel(vocab_size)
model = model.to(device)

print("Before training : ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)
print(decode(out.squeeze().tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print("After training : ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)
print(decode(out.squeeze().tolist()))

