import torch
import torch.nn as nn
import numpy as np
import json
from model import LyricsLSTM
from preprocess import load_and_tokenize, build_vocab

SEQ_LEN = 40       
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.002
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_sequences(tokens, word2idx, seq_len):
    """Create (input_sequence, target_word) pairs."""
    indices = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    X, y = [], []
    for i in range(len(indices) - seq_len):
        X.append(indices[i:i+seq_len])
        y.append(indices[i+1:i+seq_len+1])
    return torch.tensor(X), torch.tensor(y)

def train():
    tokens = load_and_tokenize('lyrics.txt')
    word2idx, idx2word = build_vocab(tokens)

    with open('vocab.json', 'w') as f:
        json.dump({'word2idx': word2idx, 'idx2word': {str(k): v for k, v in idx2word.items()}}, f)

    X, y = make_sequences(tokens, word2idx, SEQ_LEN)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LyricsLSTM(vocab_size=len(word2idx)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f'Training on {DEVICE} | Vocab: {len(word2idx)} | Sequences: {len(X)}')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output.view(-1, len(word2idx)), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), 'lyrics_model.pt')
    print('Model saved to lyrics_model.pt')

if __name__ == '__main__':
    train()