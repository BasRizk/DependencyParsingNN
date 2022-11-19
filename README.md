# DependencyParsingNN
Replicating partly A Fast and Accurate Dependency Parser Using Neural Networks’ by Danqi Chen and Chris Manning and conducting few expirements 


## `prepare_data.py`
Converts CoNLL data (train and dev) into features of the parser configuration paired with parser decisions, takes in a dependency tree and, using SHIFT-REDUCE-PARSING, determining parser actions, which will alter the parser configuration, from which the feature set can be determined.

### Parameters:
  - `-f` data files (default: `train.orig.conll dev.orig.conll`)

### Format of generated files
#### (filename format: `WORD_BEFORE_DOT.converted`)
prepare_data.py puts the data into csv `WORD_BEFORE_DOT.converted` file with the following columns:
>     [
>       's_1', 's_2', 's_3',
>       'b_1', 'b_2', 'b_3',
>       'lc_1(s_1)', 'rc_1(s_1)', 'lc_2(s_1)', 'rc_2(s_1)',
>       'lc_1(s_2)', 'rc_1(s_2)', 'lc_2(s_2)', 'rc_2(s_2)',
>       'lc_1(lc_1(s_1))', 'rc_1(rc_1(s_1))',
>       'lc_1(lc_1(s_2))', 'rc_1(rc_1(s_2))'
>     ]

where given a sentence:
-  `s_i` corresponds to element (token) `i` on its stack,
-  `b_i` corresponds to element i on its buffer,
-  `lc_i(x)` corresponds to `ith` left child of element `x`
-  `rc_i(x)` corresponds to `ith` right child of element `x`

## `train.py`
train.py trains a model given data preprocessed by preparedata.py and writes a model file train.model, including vocab data.

### Parameters:
  - `-t` training file (default: `train.converted`)
  - `-d` validation (dev) fiile (default: `dev.converted`)
  - `-E` word embedding dimension (default: `50`)
  - `-e` number of epochs (default: `10`) 
  - `-u` number of hidden units (default: `200`)
  - `-lr` learning rate (default: `0.01`)
  - `-reg` regularization amount (default: `1e-5`)
  - `-batch` mini-batch size (default: `256`)
  - `-o` model filepath to be written (default: `train.model`)
  - `-gpu` use gpu (default: `True`)


## `parse.py`:
Given a trained model file (and possibly vocabulary file reads in CoNLL data and writes CoNLL data where fields 7 and 8 contain dependency tree info.

### Parameters:
  - `-m` model filepath (default: `train.model`)
  - `-i` input CoNLL filepath (deault: `parse.in`)
  - `-o` output CoNLL filepath (default: `parse.out`)
  - `-verbose` show progress bar (default: `False`)
  


### Example
>
> EXEC_FILE = train.py
> or
> EXEC_FILE = train-torch.py 
> 
> `python $EXEC_FILE -u $HIDDEN_UNITS -l $LEARNING_RATE -f $MAX_SEQUENCE_LENGTH -b $MINI_BATCH_SIZE -e $NUM_EPOCHS -E $EMBEDDING_FILE -i $DATASET -o $OUT_MODEL_FILE -w $WEIGHTS_INIT -d $DEBUG_FILE`



## Instructions for Classifying
### Parameters:
  - `-m` model filename (either start with `pytorch` or without)
  - `-i` test data-set relative filepath
  - `-o` output (inference) desired relative filepath
  
  
### Example
>
> EXEC_FILE = train.py
> or
> EXEC_FILE = train-torch.py 
> 
> `python $EXEC_FILE -m nb.4dim.model -i 4dim.sample.txt -o 4dim.out.txt`
> 

<!-- ## Ideas for ‘Extra Mile’ work
This list is not exhaustive. As always, better analysis of the things you try and justification for what you
try will lead to more credit.
• As noted above, modify the features
• Try building an arc-eager model. Do the results differ much, in terms of LAS/UAS, run time, training
time, etc?
• Try different sets of external word embeddings, including the Collobert ones Danqi used and better
ones that have come out since.
• Try to perfectly replicate Danqi’s work: use a cubic nonlinear function, use external embeddings, use
POS tagger-generated tags, etc.
• Compare different (reasonable) hyperparameter/learning configuration variants; hidden unit size, number of layers, learning rates, gradient descent variants and schedules, etc.
• Ablate feature sets, introduce other new (meaningful) features
• Look for papers that cited Danqi’s paper. See if they came up with any advances and try implementing
them. -->
