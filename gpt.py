import torch
import torch.nn as nn
from torch.nn import functional as F

import os
os.environ["CUDA_VISIBLE_DEVICE"] = "1"

# hyperparameters
block_size = 256
batch_size = 64
max_iters = 10_000
eval_interval = 500
learning_rate = 1e-5
eval_iters = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

n_embed = 384
n_layers = 6
n_heads = 6
dropout = 0.2
# ------------------
data_file = "input.txt"
model_file = "models/gpt.pth"
checkpt_file = "checkpoints/gpt.tar"

os.makedirs(os.path.dirname(model_file), exist_ok=True)
os.makedirs(os.path.dirname(checkpt_file), exist_ok=True)



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


# # BEBUG: Testing
# n_embed = 4
# batch_size = 4
# block_size = 5
# 
# def test_modules(model):
#     tok_table = nn.Embedding(100, n_embed, device=device)
#     
#     x, y = get_batch("train")
#     embed = tok_table(x)
#     out = model(embed)
#     
#     print("Output matrix : ")
#     print(out)
#     print(f"Ouput size : {out.shape}")
# 
# model = FeedForward()
# model = model.to(device)
# test_modules(model)
# exit()
# # BEBUG: Testing


class Head(nn.Module):

    def __init__(self, head_size):
        super(Head, self).__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embed):
        k = self.key(embed)   # B,T,n_hd
        q = self.query(embed) # B,T,n_hd

        B, T, H = k.shape
        weight = q @ k.transpose(-1, -2)*(H**-0.5) # B,T,T
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1) # B, T, T
        weight = self.dropout(weight)

        v = self.value(embed)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embed):
        out = torch.cat([h(embed) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):

    def __init__(self):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, embed):
        return self.net(embed)


class Block(nn.Module):

    def __init__(self, num_heads):
        super(Block, self).__init__()

        head_size = n_embed//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(GPTLanguageModel, self).__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx) # (B,T,C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        embed = tok_embed + pos_embed

        x = self.blocks(embed)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(-1))

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx




model = GPTLanguageModel(vocab_size)
model = model.to(device)

print("Before training : ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)
print(decode(out.squeeze().tolist()))
print("------------------------------", end="\n\n")

# Model training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Saving checkpoints
        torch.save({ 
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "step": i
                    }, checkpt_file)

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Model saving
torch.save({ 
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "losses": losses,
            }, model_file)


print("After training : ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)
print(decode(out.squeeze().tolist()))
print("------------------------------", end="\n\n")


