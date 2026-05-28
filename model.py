import torch
import torch.nn as nn

class LyricsLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(LyricsLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embeds, hidden)
        output = self.fc(self.dropout(lstm_out))
        return output, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(2, batch_size, 256).to(device),
            torch.zeros(2, batch_size, 256).to(device)
        )