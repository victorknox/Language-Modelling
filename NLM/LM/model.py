import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.embedding_dim = 100
        self.num_layers = 1

        self.n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers )
        self.fc = nn.Linear(self.lstm_size, self.n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(device), torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(device))