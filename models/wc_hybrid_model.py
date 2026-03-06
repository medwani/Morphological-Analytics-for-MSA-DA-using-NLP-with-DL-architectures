import torch
import torch.nn as nn

class WC_Hybrid(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, word_embedding_dim, char_embedding_dim, 
                 hidden_dim, num_tags):
        super(WC_Hybrid, self).__init__()
        # Word-level part
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.word_bilstm = nn.LSTM(word_embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Char-level part
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_cnn = nn.Conv1d(char_embedding_dim, hidden_dim, kernel_size=3, padding=1)

        # Combined
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim, num_tags)

    def forward(self, word_input, char_input):
        # Word processing
        word_embedded = self.word_embedding(word_input)
        word_out, _ = self.word_bilstm(word_embedded)

        # Character processing
        char_embedded = self.char_embedding(char_input)
        char_embedded = char_embedded.permute(0, 2, 1) # Conv1d expects (batch, channels, seq_len)
        char_out = self.char_cnn(char_embedded)
        char_out = char_out.permute(0, 2, 1) # Back to (batch, seq_len, channels)
        
        # For simplicity, we are not aligning the outputs of word and char models perfectly here.
        # A real implementation would require more careful handling of sequence lengths.
        # Here we assume they can be concatenated directly.
        combined = torch.cat((word_out, char_out), dim=2)
        
        out = self.fc(combined)
        return out
