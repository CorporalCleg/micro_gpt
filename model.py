import torch
import torch.nn as nn
import torch.nn.functional as F
from v2 import *


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

with open('mgpt/input.txt', 'r', encoding='utf-8') as f:#load text
    text = f.read()

chars = sorted(list(set(text)))#count chars
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}#string take output integer
itos = {i:ch for i, ch in enumerate(chars)}#think yourself

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda seq: ''.join(itos[num] for num in seq)

idx = torch.zeros((1, 1), dtype=torch.long).to(device)

model = BigramLanguageModel()
model = torch.load('mgpt/shakespeare+.pt')

print(decode(model.generate(idx, max_new_token=5000)[0].cpu().tolist()))