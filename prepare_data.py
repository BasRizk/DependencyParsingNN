# preparedata.py converts CoNLL data (train and dev) into features of the parser configuration paired
# with parser decisions. This should be human-readable, i.e. a text file of words/labels. 
# TODO The format should be described in the README
# preparedata.py should take in a dependency tree and, based on the heuristics discussed in class,
# determine parser actions, which will alter the parser configuration, from which the feature set can
# be determined. Meanwhile 

#################################################################################
# Each line is in CoNLL data format, and has 10 tab-separated fields as follows:
# 1. current sentence word number (1-based)
# 2. the word
# 3. lemma form of the word
# 4. coarse-grained POS tag
# 5. fine-grained POS tag (in practice, both of these are the same)
# 6. morphological features (in practice, just the token ‘-’)
# 7. token id of this word’s parent (head),
# or 0 if this word’s parent is the root. 
# If this data has not yet been provided 
# (i.e. the sentence isn’t parsed) it is ’-’.
# 8. dependency label for the relation of this word to its parent.
# If this data has not yet been provided (i.e.
# the sentence isn’t parsed) it is ’-’.
# 9. other dependencies (in practice, just ‘-’)
# 10. other info (in practice, just ‘-’)
#################################################################################


import pandas as pd
from data_utils import FeatureGenerator, Sentence, Token

def read_parse_tree(filepath="train.orig.conll"):
    def split_conll_line(file):
        line =  file.readline().strip()
        if line:
            return line.split("\t")
        return line
    
    with open(filepath) as parseTreesFile:
        sentences = []
        while True:
            line = split_conll_line(parseTreesFile)
            if not line:
                # File is over
                break
            sentence_tokens = []
            while(True):
                token = Token(
                    token_id = line[0],
                    word = line[1],
                    pos = line[3],
                    head = line[6],
                    dep = line[7],
                )
                sentence_tokens.append(token)
                line = split_conll_line(parseTreesFile)
                if not line:
                    # Sentence is over
                    sentences.append(
                        Sentence(sentence_tokens)
                    )
                    break
                
    return sentences

if __name__ == "__main__":
    features_generator = FeatureGenerator()
    data_files = ["train.orig.conll", "dev.orig.conll"]
    for filepath in data_files:
        sentences = read_parse_tree(filepath)
        print(f'Finished reading {filepath} file')
        train_dataset =\
            features_generator.generate_dataset(
                sentences
                )
        print("Finished generating features") 
        out_filepath = f'{filepath.split(".")[0]}.converted'
        train_dataset.to_csv(out_filepath, index=False)
        print(f"Written {out_filepath}")        