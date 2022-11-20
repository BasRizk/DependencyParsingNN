
import numpy as np
import pandas as pd
from data_utils import UNK

class BasicWordEncoder:
    def __init__(self, training_df) -> None:
        unique_words = pd.unique(training_df[training_df.columns[:-1]].values.ravel('K'))
        self.word_to_idx = {w: i+1 for i, w in enumerate(unique_words)}
        self.word_to_idx[UNK] = 0
        self.unique_labels = pd.unique(training_df[training_df.columns[-1]])
        self.label_to_idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.label_to_idx[UNK] = 0

    def _encode_word(self, word):
        idx = self.word_to_idx.get(word)
        if idx is None:
            idx = self.word_to_idx[UNK]
        return idx
    
    def _encode_label(self, label):
        idx = self.label_to_idx.get(label)
        if idx is None:
            idx = self.label_to_idx[UNK]
        return idx
        
    def decode_label(self, label_embed):
        return self.unique_labels[label_embed]
    
    def encode_dataset(self, df, labeled=True):
        if labeled:
            df[df.columns[:-1]] = df[df.columns[:-1]].applymap(lambda x: self._encode_word(x))
            df[df.columns[-1:]] = df[df.columns[-1:]].applymap(lambda x: self._encode_label(x))
        else:
            df = df.applymap(lambda x: self._encode_word(x))
            df = df.applymap(lambda x: self._encode_label(x))
        return df
    
    def encode_features_vector(self, feats):
        return np.vectorize(self._encode_word)(feats)

    def get_dictionary_size(self):
        return len(self.word_to_idx)
    
    def get_num_of_labels(self):
        return len(self.unique_labels)