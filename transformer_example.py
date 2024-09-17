import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Hyperparameters
batch_size = 64
seq_length = 20    # Number of tokens per sequence
embedding_size = 128
nhead = 8
d_model = 128
num_layers = 2
lr = 0.0005
epochs = 20

# Sample Text Data
sample_text = """
Once upon a time, in a land far, far away, there was a kingdom where magic thrived.
The people lived in harmony with the mystical creatures that roamed the forests and mountains.
Among them was a young wizard named Arin, who sought to master the ancient arts.
He ventured into the deepest parts of the enchanted woods, discovering secrets lost to time.
With each passing day, his power grew, and so did his understanding of the world's mysteries.
"""

# Preprocessing: Tokenization and Vocabulary Building
def tokenize(text):
    text = text.lower()
    tokens = text.split()
    return tokens

tokens = tokenize(sample_text)
vocab = set(tokens)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

# Create input and target sequences
def create_sequences(tokens, seq_length):
    input_seqs = []
    target_seqs = []
    for i in range(len(tokens) - seq_length):
        input_seq = tokens[i:i+seq_length]
        target_seq = tokens[i+1:i+seq_length+1]
        input_seqs.append([word_to_idx[word] for word in input_seq])
        target_seqs.append([word_to_idx[word] for word in target_seq])
    return input_seqs, target_seqs

input_seqs, target_seqs = create_sequences(tokens, seq_length)

# Convert to tensors
input_seqs = torch.tensor(input_seqs, dtype=torch.long)
target_seqs = torch.tensor(target_seqs, dtype=torch.long)

# DataLoader
dataset = torch.utils.data.TensorDataset(input_seqs, target_seqs)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(tgt):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(tgt)).to(device)
            self.src_mask = mask

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask=self.src_mask)
        output = self.fc_out(output)
        return output

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.pe = pe.unsqueeze(1)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Initialize Model, Loss Function, Optimizer
model = TransformerModel(vocab_size, d_model, nhead, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in data_loader:
        src, tgt = batch  # src and tgt shape: [batch_size, seq_length]
        src = src.transpose(0, 1)  # Shape: [seq_length, batch_size]
        tgt = tgt.transpose(0, 1)  # Shape: [seq_length, batch_size]

        tgt_input = tgt[:-1, :]  # Shape: [seq_length - 1, batch_size]
        tgt_output = tgt[1:, :].reshape(-1)  # Shape: [(seq_length - 1) * batch_size]

        optimizer.zero_grad()
        output = model(src, tgt_input)  # Output shape: [seq_length - 1, batch_size, vocab_size]
        loss = criterion(output.view(-1, vocab_size), tgt_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')


# Text Generation
def generate_text(model, start_words, num_generate=20):
    model.eval()
    words = start_words.lower().split()
    input_ids = [word_to_idx.get(word, 0) for word in words]
    src = torch.tensor([input_ids], dtype=torch.long).transpose(0, 1)
    generated = words.copy()
    with torch.no_grad():
        for _ in range(num_generate):
            tgt_input = src[-seq_length:]
            output = model(src, tgt_input)
            next_token_logits = output[-1, 0, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_word = idx_to_word.get(next_token_id, '<unk>')
            generated.append(next_word)
            next_input = torch.tensor([[next_token_id]], dtype=torch.long)
            src = torch.cat([src, next_input], dim=0)
    return ' '.join(generated)

# Generate Text Starting with "Once upon a time"
start_words = "Once upon a time"
generated_text = generate_text(model, start_words)
print("\nGenerated Text:")
print(generated_text)
