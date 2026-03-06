import torch
import torch.nn as nn

class CL_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, num_tags):
        super(CL_BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                              bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags) # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        out = self.fc(lstm_out)
        return out
