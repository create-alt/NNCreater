import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.model(x)
