import pickle
import io
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from data_utils import BasicWordEncoder

torch.manual_seed(1)

class LabeledDataset(Dataset):

    def __init__(self, df):
        x = df[df.columns[:-1]]
        y = df[df.columns[-1:]]
        self.x = torch.tensor(x.values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx].squeeze()
    
class NeuralNetwork(torch.nn.Module):

    def __init__(self, dictionary_size, embedding_dim,
                 num_of_tokens, num_transitions,
                 hidden_size,
                 embedding_weight=None) -> None:
        super().__init__()
        
        if embedding_weight is not None:
            embedding_weight = torch.randn((dictionary_size, embedding_dim))*embedding_weight

        self.dictionary_size = dictionary_size
        self.num_of_tokens = num_of_tokens
        self.num_transitions = num_transitions
        self.embeddings = nn.Embedding(
            dictionary_size, embedding_dim, _weight=embedding_weight
        )
        self.linear1 = nn.Linear(num_of_tokens * embedding_dim, hidden_size)
        self.activation = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(hidden_size, num_transitions)

    def forward(self, x):
        embeds = self.embeddings(x).view((x.shape[0], -1))
        out = self.activation(self.linear1(embeds))
        out = self.linear2(out)
        return out


#########################################################################
# CPU_Unpickler taken from:
# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
#########################################################################
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)
    
class Model:
    def __init__(self,
            word_encoder: BasicWordEncoder, embedding_dim, num_of_tokens,
            hidden_size, learning_rate, regularization_rate,
            embedding_weight=None,
            use_gpu=True) -> None:
        
        self.use_gpu = use_gpu
        self.word_encoder = word_encoder
        self.network = NeuralNetwork(
            word_encoder.get_dictionary_size(),
            embedding_dim,
            num_of_tokens,
            word_encoder.get_num_of_labels(),
            hidden_size,
            embedding_weight=embedding_weight
        )
        self.init_device()
        
        self.optimizer = torch.optim.Adagrad(
            self.network.parameters(), lr=learning_rate, lr_decay=0,
            weight_decay=regularization_rate,
            initial_accumulator_value=0, eps=1e-10
        )

        self.criterion = nn.CrossEntropyLoss()
    
        print(self.network)
        
    def init_device(self):
        if self.use_gpu and torch.cuda.is_available():
            print('Using GPU')
            self.device = torch.device('cuda')
        else:
            print('Using CPU')
            self.device = torch.device('cpu')   
        self.network.to(self.device)

    def debug(self, statement='', end="\n", flush=True):
        if self.debug_file:
            self.debug_file.write(str(statement) + end)
        print(statement, end=end, flush=flush)

    def train_model(self,
        train_loader: DataLoader, dev_loader: DataLoader,
        epochs: int, verbose=1):

        epochs_trange = tqdm(range(epochs), desc='Epochs')
        
        for _ in epochs_trange:
            train_loss, train_acc =\
                self._train_epoch(train_loader, epochs_trange)            
        
            val_loss, val_acc = self.evaluate(dev_loader)
        
            epochs_trange.set_postfix({
                'loss': f'{train_loss:5.2f}',
                'acc' : f'{train_acc: < .2f}',
                'vloss': f'{val_loss:5.2f}',
                'vacc' : f'{val_acc: < .2f}'
            })
            print()
            
            
    def _train_epoch(self, train_loader: DataLoader, epochs_trange: tqdm):
        num_batches = len(train_loader)
        accumulated_training_loss = 0
        training_corrects = 0
        
        self.network.train() # turn on training mode
        
        # loop over the data iterator, and feed the inputs to the network and adjust the weights.
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            preds = self.network(X)
            loss = self.criterion(preds, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            training_corrects += (preds.argmax(1) == y).sum().item()
            accumulated_training_loss += loss.item()

            epochs_trange.set_postfix({
                'batch' : f'{batch_idx + 1:5d}/{num_batches:5d} ',
                'loss': f'{accumulated_training_loss/(batch_idx+1): < .2f}',
                'acc' : f'{training_corrects/((batch_idx+1)*X.shape[0]): < .2f}'
            })
        return accumulated_training_loss/(batch_idx+1), training_corrects/len(train_loader.dataset)
            
                            
    def evaluate(self, dev_loader: DataLoader) -> float:
        self.network.eval()  # turn on evaluation mode
        total_loss = 0.
        corrects = 0
        with torch.no_grad():
            for _, (X, y) in enumerate(dev_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                preds = self.network(X)
                                
                total_loss += self.criterion(preds, y).item()
                corrects += (preds.argmax(1) == y).sum().item()
        return total_loss / (len(dev_loader) - 1), corrects/len(dev_loader.dataset)
    
    def save_model(self, model_filename):
        with open(model_filename, "wb") as file:
            self.device = torch.device('cpu')
            self.network.to(self.device)
            print('Transfered forcibly model to CPU')
            pickle.dump(self, file)

    @staticmethod
    def load_model(model_file):
        # static so can be called as Model.load_model('examplemodel.model')
        with open(model_file, "rb") as file:
            # model = pickle.load(file)
            model = CPU_Unpickler(file).load()
            model.init_device()
            model.network.to(model.device)
        return model
    
    def classify(self, x, pre_encoded=False):
        if not pre_encoded:
            x = self.word_encoder.encode_features_vector(x)
        x = torch.tensor(x, dtype=torch.long)
        x = x.to(self.device)
        pred =  self.network(x).argmax(1).item()
        decoded_pred = self.word_encoder.decode_label(pred)
        return decoded_pred
        
