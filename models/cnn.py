import torch
import torch.nn as nn
from torchvision import models

class CNNBaseline(nn.Module):
    def __init__(self, vocab_size, num_answers, dropout=0.5):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.visual = nn.Sequential(*list(resnet.children())[:-1])

        self.embed = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.lstm = nn.LSTM(
            300, 256, num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_answers),
        )

    def forward(self, images, questions, lengths):
        v = self.visual(images).flatten(1)
        q = self.embed(questions)
        packed = nn.utils.rnn.pack_padded_sequence(
            q, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        q = torch.cat([h[-2], h[-1]], dim=1)
        return self.classifier(torch.cat([v, q], dim=1))
