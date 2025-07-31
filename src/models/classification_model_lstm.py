"""
This file implements a classification model LSTM.
"""

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=128):
        super(LSTMClassifier, self).__init__()

        #batch_first = True means tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])
