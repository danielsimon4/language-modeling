import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64       # number of chunks per bacth
block_size = 256      # chunks maximum context length
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200      # how many iterations used to calculate the loss
eval_interval = 500   # every how many iterations calculate the loss
learning_rate = 3e-4
max_iters = 5000
n_embd = 384          # number of embedding dimensions
n_layer = 6           # number of blocks
n_head = 6            # number of heads
dropout = 0.2         # dropout percentage


# ----------------------------------------------------------------------------------------------


torch.manual_seed(1337)

# load dataset
with open('NanoGPT/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters and vocabulary size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]

# decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n] # 90%
val_data = data[n:]   # 10%

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # put the model in evaluation mode

    # calculate train loss and evaulation loss
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train() # put the model back in train mode
    return out


# ----------------------------------------------------------------------------------------------


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        _, T, _ = x.shape # (B, T, n_embd)

        # compute keys and queries
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, T) = (B, T, hs) @ (B, hs, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = wei @ v     # (B, T, hs) = (B, T, T) @ (B, T, hs)

        return out


# ----------------------------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # multi-headed self-attention
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # projection and dropout layers
        out = self.dropout(self.proj(out))
        
        return out


# ----------------------------------------------------------------------------------------------


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # growing inner-layer
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------------------------


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # communication with residual connections and layer norm
        x = x + self.sa(self.ln1(x))

        # computation with residual connections and layer norm
        x = x + self.ffwd(self.ln2(x))

        return x

# ----------------------------------------------------------------------------------------------


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # identity and positional embeddings from lookup tables
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)

        # apply blocks
        x = self.blocks(x)    # (B, T, n_embd)

        # final layer norm
        x = self.ln_f(x)      # (B, T, n_embd)

        # logits from language modeling head
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is the current context
        # in each iteration idx will grow:
        # (B, T), (B, T+1), (B, T+2), ..., (B, T+max_new_tokens)

        for _ in range(max_new_tokens):

            # crop the context to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get logits for current context (calling forward method)
            logits, _ = self(idx_cond) # (B, T, C)

            # focus only on the last time-step because
            # those are the predictions for what comes next
            logits = logits[:, -1, :] # (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx


# ----------------------------------------------------------------------------------------------


# intialize the model
model = GPTLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# train the model
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step: {iter:5d}/{max_iters:5d}   Train loss: {losses['train']:.4f}   Val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # update
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=700)[0].tolist()))