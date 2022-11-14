# TODO • train.py trains a model given data preprocessed by preparedata.py and writes a model file
# 
# TODO train.model is the model file (also possibly vocab file named train.vocab), the result of running
# train.py on training and dev with base feature set.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

torch.manual_seed(1)

class Model(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_transitions) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, num_transitions)


    def forward(self, x):
        embeds = self.embeddings(x).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# • Many of the features will be empty values (e.g. the leftmost child of the leftmost child of the second
# word on the stack...when the stack contains one item). This is a feed-forward network, though, so some
# placeholder None value should appear in these cases.


def train_model(model):
    pass

if __name__ == "__main__":
    # Read training/dev data
    training_df = pd.read_csv("train.converted", )
    dev_df = pd.read_csv("dev.converted")
    breakpoint()
    # Encode data
    # context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

    # Create model
    # Train model
    