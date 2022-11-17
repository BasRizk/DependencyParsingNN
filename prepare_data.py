from data_utils import FeatureGenerator
from data_parser import DataParser

"""_summary_
preparedata.py converts CoNLL data (train and dev) into features of the parser configuration paired
with parser decisions. This should be human-readable, i.e. a text file of words/labels. 
TODO The format should be described in the README
preparedata.py should take in a dependency tree and, based on the heuristics discussed in class,
determine parser actions, which will alter the parser configuration, from which the feature set can
be determined.
"""

if __name__ == "__main__":
    features_generator = FeatureGenerator()
    data_files = ["train.orig.conll", "dev.orig.conll"]
    for filepath in data_files:
        sentences = DataParser.read_parse_tree(filepath)
        print(f'Finished reading {filepath} file')
        train_dataset =\
            features_generator.generate_labeled_dataset(
                sentences
                )
        print("Finished generating features") 
        out_filepath = f'{filepath.split(".")[0]}.converted'
        train_dataset.to_csv(out_filepath, index=False)
        print(f"Written {out_filepath}")        
      