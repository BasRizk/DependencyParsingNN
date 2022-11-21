import time
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from word_encoder import BasicWordEncoder
from model import Model, LabeledDataset
"""_summary_
train.py trains a model given data preprocessed by preparedata.py and writes a model file train.model, including vocab data.
"""
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
    parser.add_argument('-o',  default='train.model', type=str, help='model file to be written')
    parser.add_argument('-emb_w_init',  default=1, type=float, help='embedding weights scaling')
    parser.add_argument('-gpu', default=True, type=bool, help='use gpu')
    args = parser.parse_args()
        
    # Read training/dev data
    train_df = pd.read_csv(args.t)
    dev_df = pd.read_csv(args.d)
    print('Finished Reading Dataset')
    
    # Encode data
    word_encoder = BasicWordEncoder(train_df)
    print(f'Encoding based on {word_encoder.get_dictionary_size()} vocab '
          f'& {word_encoder.get_num_of_labels()} labels')
    encode_stime = time.time()
    train_df = word_encoder.encode_dataset(train_df)
    print(f'Encoding Train data in {time.time() - encode_stime: < .2f}s')
    encode_stime = time.time()
    dev_df = word_encoder.encode_dataset(dev_df)
    print(f'Encoding Dev data in {time.time() - encode_stime: < .2f}s')
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
        embedding_weight=args.emb_w_init,
        use_gpu=args.gpu
    )
    print('Finished Building Model')

    # Train model
    train_stime = time.time()
    model.train_model(train_loader, dev_loader, epochs=args.e)
    print(f'Finished Training model in {time.time() - train_stime: < .2f}s')

    # Saving model
    model.save_model(args.o)
    print(f'Finished saving model as {args.o}')
    