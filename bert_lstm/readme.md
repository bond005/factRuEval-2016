# BERT embeddings as features for Russian NER

This program is devoted for experiments with named entity recognition on the FactRuEval-2016 dataset using the BERT as subword features.

BERT embeddings for different languages are described in this document https://github.com/google-research/bert/blob/master/multilingual.md

For finally classification of named entities we use some algorithms: CRF (conditional random field), BiLSTM (bidirectional LSTM neural network with single LSTM layer), hybrid BiLSTM-CRF network and randomly classifier as naive baseline.

Example of using:

```
python create_submit.py -m model/lstm -d cached_data/data -r /path/to/directory/with/submit/results -t bilstm -n 100
```

