# ELMo embeddings as features for Russian NER

This program is devoted for experiments with named entity recognition on the FactRuEval-2016 dataset using the DeepPavlov ELMo as word features.

ELMo embeddings for the Russian language are described in this manual http://docs.deeppavlov.ai/en/master/apiref/models/embedders.html#deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder

For finally classification of named entities we use some algorithms: CRF (conditional random field), BiLSTM (bidirectional LSTM neural network with single LSTM layer), hybrid BiLSTM-CRF network and randomly classifier as naive baseline.

Example of using:

```
python create_submit.py -m model/lstm -d cached_data/data -r /path/to/directory/with/submit/results -t bilstm -n 100
```

F1-measure with DeepPavlov ELMO embeddings for the Russian language and simple bidirectional LSTM network (number of neurons in the LSTM layer is 512, dropout is 0.7, recurrent dropout is 0.0) is 88.47%. And if we use more simpler model, CRF, with these ELMo, then our F1-score increases to 89.01%.
