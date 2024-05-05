import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1110)

#hyperparameters
batch_size = 64
block_size = 128
max_iters = 8000
eval_interval = max_iters // 10
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192
n_head = 6
n_layers = 3
drpt = 0.2
# ------------


with open('comp/ala_gpt/input.txt', 'r', encoding='utf-8') as f:#load text
    text = f.read()

chars = sorted(list(set(text)))#count chars
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}#string take output integer
itos = {i:ch for i, ch in enumerate(chars)}#think yourself

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda seq: ''.join(itos[num] for num in seq)

#encode data
data = torch.tensor(encode(text), dtype=torch.long)

#apply split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)

    return x, y

class Head(nn.Module):

    '''
    singe self-attention block
    '''
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=drpt)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        self.dropout(x)
        v = self.value(x)
        out = wei @ v
        return out

# class Head(nn.Module):
#     """ one head of self-attention """

class MultiHeadAttention(nn.Module):
    '''
    smultiple head attention block
    '''

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_heads))
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim =-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(p=drpt)
        )
    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads=n_heads, head_size=n_embd // n_heads)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)#make vector from token
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(*[Block(n_embd=n_embd, n_heads=n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.ln(self.block(x))
        logits = self.lm_head(x)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_cond = idx[:, -block_size:]
            #make prediction
            logits, _ = self(idx_cond)
            #take last element
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

@torch.no_grad()
def esimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



if __name__ == '__main__':
    model = BigramLanguageModel().to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)


    for iter in range(max_iters):

        if iter % eval_interval == 0:
            out = esimate_loss()
            print(f"step {iter}: train loss {out['train']:.4f}, eval loss {out['val']:.4f}")
            torch.save(model, 'shakesapre+.pt')

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx = torch.zeros((1, 1), dtype=torch.long)

    torch.save(model, 'shakespeare.pt')
    print(decode(model.generate(idx, max_new_token=200)[0].cpu().tolist()))