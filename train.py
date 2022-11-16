# TODO • train.py trains a model given data preprocessed by preparedata.py and writes a model file
#
# TODO train.model is the model file (also possibly vocab file named train.vocab), the result of running
# train.py on training and dev with base feature set.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import pandas as pd
import argparse
from data_utils import BasicWordEncoder

torch.manual_seed(1)


class NeuralNetwork(torch.nn.Module):

    def __init__(self, dictionary_size, embedding_dim,
                 num_of_tokens, num_transitions,
                 hidden_size) -> None:
        super().__init__()
        self.dictionary_size = dictionary_size
        self.num_of_tokens = num_of_tokens
        self.num_transitions = num_transitions
        self.embeddings = nn.Embedding(dictionary_size, embedding_dim)
        self.linear1 = nn.Linear(num_of_tokens * embedding_dim, hidden_size)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_size, num_transitions)

    def forward(self, x):
        embeds = self.embeddings(x).view((x.shape[0], -1))
        out = self.activation(self.linear1(embeds))
        out = self.linear2(out)
        return out


class LabeledDataset(Dataset):

    def __init__(self, df):
        x = df[df.columns[:-1]]
        y = df[df.columns[-1:]]
        self.x_train = torch.tensor(x.values, dtype=torch.long)
        self.y_train = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

# • Many of the features will be empty values (e.g. the leftmost child of the leftmost child of the second
# word on the stack...when the stack contains one item). This is a feed-forward network, though, so some
# placeholder None value should appear in these cases.


class Model:
    def __init__(self,
               dictionary_size, embedding_dim, num_of_tokens,
               num_transitions, hidden_size,
               learning_rate,
               regularization_rate) -> None:
        self.network = NeuralNetwork(
            dictionary_size,
            embedding_dim,
            num_of_tokens,
            num_transitions,
            hidden_size
        )

        self.optimizer = torch.optim.Adagrad(
            self.network.parameters(), lr=learning_rate, lr_decay=0,
            weight_decay=regularization_rate,
            initial_accumulator_value=0, eps=1e-10
        )

        self.loss_fn = nn.CrossEntropyLoss()
        
        print(self.network)
        

    def debug(self, statement='', end="\n", flush=True):
        if self.debug_file:
            self.debug_file.write(str(statement) + end)
        print(statement, end=end, flush=flush)

    def train_model(self,
        train_loader: DataLoader, dev_loader: DataLoader,
        epochs: int, verbose=1):

        epochs_trange = tqdm(range(epochs), desc='Epochs')
        
        for epoch in epochs_trange:
            epoch_start_time = time.time()
            self._train_epoch(train_loader, epochs_trange, verbose=verbose)
            
            val_loss = self.evaluate(dev_loader)
            
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f}')
            print('-' * 89)
            
            
    def _train_epoch(self, train_loader, epoch, verbose, log_interval=200):
        
        
        num_batches = len(train_loader)
        total_loss = 0

        self.network.train() # turn on training mode
        
        # loop over the data iterator, and feed the inputs to the network and adjust the weights.
        for batch_idx, (X, y) in enumerate(train_loader):
            preds = self.network(X)
            loss = self.loss_fn(preds, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accumulated_training_loss += loss
            accumulated_training_corrects +=\
                (preds.argmax(1) == torch.argmax(y, axis=1)).type(
                    torch.float).sum().item()
            total_loss += loss.item()
            
            # if verbose > 0:
            #     epochs_trange.set_postfix({
            #         'batch': f'{batch_idx+1}/{num_of_batches}',
            #         'loss': f'{accumulated_training_loss/(batch_idx+1): < .2f}'
            #     })
            
            total_loss += loss.item()
            if batch_idx % log_interval == 0 and batch_idx > 0:
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = torch.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch_idx:5d}/{num_batches:5d} batches | '
                    f'ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

    def evaluate(self, dev_loader: DataLoader) -> float:
        self.network.eval()  # turn on evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for i, (X, y) in enumerate(dev_loader):
                preds = self.network(X)
                total_loss += self.loss_fn(preds, y)
                
        return total_loss / (len(dev_loader) - 1)
          
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Neural net training arguments.')
    parser.add_argument('-u', default=200, type=str, help='number of hidden units')
    parser.add_argument('-l', default=0.01, type=float, help='learning rate')
    parser.add_argument('-r', default=1e-5, type=float, help='regularization amount')
    parser.add_argument('-b', default=32, type=int, help='mini-batch size')
    parser.add_argument('-e', default=100, type=int, help='number of epochs to train for')
    parser.add_argument('-E', default=50, type=int, help='word embedding dimension')
    parser.add_argument('-t', default="train.converted", type=str, help='training file')
    parser.add_argument('-d', default="dev.converted", type=str, help='validation (dev) file')
    parser.add_argument('-o', type=str, help='model file to be written')
    args = parser.parse_args()
    
    # Read training/dev data
    train_df = pd.read_csv(args.t)
    dev_df = pd.read_csv(args.d)
    print('Finished Reading Dataset')
    
    # Encode data
    word_encoder = BasicWordEncoder(train_df)
    train_df = word_encoder.encode_dataset(train_df)
    print('Encoding Train data', end='\r')
    dev_df = word_encoder.encode_dataset(dev_df)
    print('Encoding Dev data', end='\r')
    print('Finished Encoding Dataset')
    
    # build data loaders    
    train_loader = DataLoader(LabeledDataset(train_df), batch_size=args.b, shuffle=True)
    dev_loader = DataLoader(LabeledDataset(dev_df), batch_size=args.b, shuffle=True)
    print('Finished Building Dataloaders')
    
    # Create model
    model = Model(
      dictionary_size=word_encoder.get_dictionary_size(),
      embedding_dim=args.E,
      num_of_tokens=len(train_df.columns)-1,
      num_transitions=word_encoder.get_num_of_labels(),
      hidden_size=args.u,
      learning_rate=args.l,
      regularization_rate=args.r
    )
    print('Finished Building Model')

    # Train model
    model.train_model(train_loader, dev_loader, epochs=args.e)
    