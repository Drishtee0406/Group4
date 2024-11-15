import torch
import numpy

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Identify all unique characters and create vocabulary mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the dataset into a tensor of integer tokens
data = torch.tensor(encode(text), dtype=torch.long)

print("Unique Characters in the dataset - Tokens")
print(''.join(chars))
print("\nFirst 100 encoded characterd in the dataset\n")
print(str(data[:100].numpy().tolist()))

with open("milestone1.txt", "w") as f:
    f.write("Unique Characters in the dataset - Tokens")
    f.write(''.join(chars))
    f.write("\nFirst 100 encoded characterd in the dataset\n")
    f.write(str(data[:100].numpy().tolist()))