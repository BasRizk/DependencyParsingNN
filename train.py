# TODO • train.py trains a model given data preprocessed by preparedata.py and writes a model file
#
# TODO train.model is the model file (also possibly vocab file named train.vocab), the result of running
# train.py on training and dev with base feature set.

import argparse
import pandas as pd
from torch.utils.data import DataLoader
from data_utils import BasicWordEncoder
from model import Model, LabeledDataset
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Neural net training arguments.')
    parser.add_argument('-t', default="train.converted", type=str, help='training file')
    parser.add_argument('-d', default="dev.converted", type=str, help='validation (dev) file')
    parser.add_argument('-E', default=50, type=int, help='word embedding dimension')
    parser.add_argument('-e', default=10, type=int, help='number of epochs to train for')
    parser.add_argument('-u', default=200, type=str, help='number of hidden units')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-reg', default=1e-5, type=float, help='regularization amount')
    parser.add_argument('-batch', default=256, type=int, help='mini-batch size')
    parser.add_argument('-o', type=str, help='model file to be written')
    parser.add_argument('-gpu', default=True, type=bool, help='use gpu')
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
    train_loader = DataLoader(LabeledDataset(train_df), batch_size=args.batch, shuffle=True)
    dev_loader = DataLoader(LabeledDataset(dev_df), batch_size=args.batch, shuffle=True)
    print('Finished Building Dataloaders')
    
    # Create model
    model = Model(
        word_encoder=word_encoder,
        embedding_dim=args.E,
        num_of_tokens=len(train_df.columns)-1,
        hidden_size=args.u,
        learning_rate=args.lr,
        regularization_rate=args.reg,
        use_gpu=args.gpu
    )
    print('Finished Building Model')

    # Train model
    model.train_model(train_loader, dev_loader, epochs=args.e)
    print('Finished Training model')
    
    # Saving model
    model_filename = 'train.model'
    model.save_model(model_filename)
    print(f'Finished saving model as {model_filename}')
    