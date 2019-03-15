from argparse import ArgumentParser
import codecs
import copy
import logging
import math
import os
import pickle
import random
import tempfile
from typing import List, Tuple, Union

import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional, Dense, Masking, LSTM, TimeDistributed
from keras_contrib.layers import CRF
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf
from bert.tokenization import FullTokenizer
from bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint


logging.basicConfig(level=logging.INFO)
factrueval_logger = logging.getLogger('bert_lstm')


NAMED_ENTITIES = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
EMBEDDING_SIZE = 768
BATCH_SIZE = 16
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
POSSIBLE_MODEL_TYPES = {'crf': 1, 'bilstm': 2, 'bilstm-crf': 3, 'random': 0}


def load_document(tokens_file_name: str, spans_file_name: str,
                  objects_file_name: str) -> Tuple[List[List[Tuple[str, int, int]]], List[List[int]]]:
    texts = []
    new_text = []
    tokens_dict = dict()
    with codecs.open(tokens_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        line_idx = 1
        cur_line = fp.readline()
        while len(cur_line) > 0:
            err_msg = 'File `{0}`: line {1} is wrong!'.format(tokens_file_name, line_idx)
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                token_description = prep_line.split()
                if len(token_description) != 4:
                    raise ValueError(err_msg)
                if (not token_description[0].isdigit()) or (not token_description[1].isdigit()) or \
                        (not token_description[2].isdigit()):
                    raise ValueError(err_msg)
                token_id = int(token_description[0])
                if token_id in tokens_dict:
                    raise ValueError(err_msg)
                new_text.append(token_id)
                token_start = int(token_description[1])
                token_length = int(token_description[2])
                tokens_dict[token_id] = (token_description[-1], 'O', token_start, token_length,
                                         (len(texts), len(new_text) - 1))
            else:
                if len(new_text) == 0:
                    raise ValueError(err_msg)
                texts.append(copy.copy(new_text))
                new_text.clear()
            cur_line = fp.readline()
            line_idx += 1
    if len(new_text) > 0:
        texts.append(copy.copy(new_text))
        new_text.clear()
    spans_dict = dict()
    with codecs.open(spans_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        line_idx = 1
        cur_line = fp.readline()
        while len(cur_line) > 0:
            err_msg = 'File `{0}`: line {1} is wrong!'.format(spans_file_name, line_idx)
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                comment_pos = prep_line.find('#')
                if comment_pos < 0:
                    raise ValueError(err_msg)
                prep_line = prep_line[:comment_pos].strip()
                if len(prep_line) == 0:
                    raise ValueError(err_msg)
                span_description = prep_line.split()
                if len(span_description) != 6:
                    raise ValueError(err_msg)
                if (not span_description[0].isdigit()) or (not span_description[-1].isdigit()) or \
                        (not span_description[-2].isdigit()):
                    raise ValueError(err_msg)
                span_id = int(span_description[0])
                token_IDs = list()
                start_token_id = int(span_description[-2])
                n_tokens = int(span_description[-1])
                if (n_tokens <= 0) or (start_token_id not in tokens_dict):
                    raise ValueError(err_msg)
                text_idx = tokens_dict[start_token_id][4][0]
                token_pos_in_text = tokens_dict[start_token_id][4][1]
                for idx in range(n_tokens):
                    token_id = texts[text_idx][token_pos_in_text + idx]
                    if token_id not in tokens_dict:
                        raise ValueError(err_msg)
                    token_IDs.append(token_id)
                if span_id not in spans_dict:
                    spans_dict[span_id] = tuple(token_IDs)
            cur_line = fp.readline()
            line_idx += 1
    with codecs.open(objects_file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        line_idx = 1
        cur_line = fp.readline()
        while len(cur_line) > 0:
            err_msg = 'File `{0}`: line {1} is wrong!'.format(objects_file_name, line_idx)
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                comment_pos = prep_line.find('#')
                if comment_pos < 0:
                    raise ValueError(err_msg)
                prep_line = prep_line[:comment_pos].strip()
                if len(prep_line) == 0:
                    raise ValueError(err_msg)
                object_description = prep_line.split()
                if len(object_description) < 3:
                    raise ValueError(err_msg)
                if object_description[1] not in {'LocOrg', 'Org', 'Person', 'Location'}:
                    factrueval_logger.warning(err_msg + ' The entity `{0}` is unknown.'.format(object_description[1]))
                else:
                    span_IDs = []
                    for idx in range(2, len(object_description)):
                        if not object_description[idx].isdigit():
                            raise ValueError(err_msg)
                        span_id = int(object_description[idx])
                        if span_id not in spans_dict:
                            raise ValueError(err_msg)
                        span_IDs.append(span_id)
                    span_IDs.sort(key=lambda span_id: tokens_dict[spans_dict[span_id][0]][2])
                    token_IDs = []
                    for span_id in span_IDs:
                        start_token_id = spans_dict[span_id][0]
                        end_token_id = spans_dict[span_id][-1]
                        text_idx = tokens_dict[start_token_id][4][0]
                        token_pos_in_text = tokens_dict[start_token_id][4][1]
                        while token_pos_in_text < len(texts[text_idx]):
                            token_id = texts[text_idx][token_pos_in_text]
                            token_IDs.append(token_id)
                            if token_id == end_token_id:
                                break
                            token_pos_in_text += 1
                        if token_pos_in_text >= len(texts[text_idx]):
                            raise ValueError(err_msg)
                    if object_description[1] in {'LocOrg', 'Location'}:
                        class_label = 'LOC'
                    elif object_description[1] == 'Person':
                        class_label = 'PER'
                    else:
                        class_label = 'ORG'
                    tokens_are_used = False
                    if tokens_dict[token_IDs[0]][1] != 'O':
                        tokens_are_used = True
                    else:
                        for token_id in token_IDs[1:]:
                            if tokens_dict[token_id][1] != 'O':
                                tokens_are_used = True
                                break
                    if not tokens_are_used:
                        tokens_dict[token_IDs[0]] = (
                            tokens_dict[token_IDs[0]][0], 'B-' + class_label,
                            tokens_dict[token_IDs[0]][2], tokens_dict[token_IDs[0]][3],
                            tokens_dict[token_IDs[0]][4]
                        )
                        for token_id in token_IDs[1:]:
                            tokens_dict[token_id] = (
                                tokens_dict[token_id][0], 'I-' + class_label,
                                tokens_dict[token_id][2], tokens_dict[token_id][3],
                                tokens_dict[token_id][4]
                            )
            cur_line = fp.readline()
            line_idx += 1
    list_of_texts = []
    list_of_labels = []
    for tokens_sequence in texts:
        new_text = []
        new_labels_sequence = []
        for token_id in tokens_sequence:
            new_text.append((tokens_dict[token_id][0], tokens_dict[token_id][2], tokens_dict[token_id][3]))
            new_labels_sequence.append(NAMED_ENTITIES.index(tokens_dict[token_id][1]))
        list_of_texts.append(new_text)
        list_of_labels.append(new_labels_sequence)
    return list_of_texts, list_of_labels


def load_data_for_training(data_dir_name: str) -> Tuple[List[List[str]], List[List[int]]]:
    names_of_files = sorted(list(filter(lambda it: it.startswith('book_'), os.listdir(data_dir_name))))
    if len(names_of_files) == 0:
        raise ValueError('The directory `{0}` is empty!'.format(data_dir_name))
    if (len(names_of_files) % 6) != 0:
        raise ValueError('The directory `{0}` contains wrong data!'.format(data_dir_name))
    list_of_all_texts = []
    list_of_all_labels = []
    for idx in range(len(names_of_files) // 6):
        base_name = names_of_files[idx * 6]
        point_pos = base_name.rfind('.')
        if point_pos <= 0:
            raise ValueError('The file `{0}` has incorrect name.'.format(base_name))
        prepared_base_name = base_name[:point_pos].strip()
        if len(prepared_base_name) == 0:
            raise ValueError('The file `{0}` has incorrect name.'.format(base_name))
        tokens_file_name = os.path.join(data_dir_name, prepared_base_name + '.tokens')
        if not os.path.isfile(tokens_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(tokens_file_name))
        spans_file_name = os.path.join(data_dir_name, prepared_base_name + '.spans')
        if not os.path.isfile(spans_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(spans_file_name))
        objects_file_name = os.path.join(data_dir_name, prepared_base_name + '.objects')
        if not os.path.isfile(objects_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(objects_file_name))
        list_of_texts, list_of_labels = load_document(tokens_file_name, spans_file_name, objects_file_name)
        list_of_all_texts += [list(map(lambda it: it[0], cur_text)) for cur_text in list_of_texts]
        list_of_all_labels += list_of_labels
    return list_of_all_texts, list_of_all_labels


def load_data_for_testing(data_dir_name: str) -> Tuple[List[List[str]], List[Tuple[str, List[List[Tuple[int, int]]]]]]:
    names_of_files = sorted(list(filter(lambda it: it.startswith('book_'), os.listdir(data_dir_name))))
    if len(names_of_files) == 0:
        raise ValueError('The directory `{0}` is empty!'.format(data_dir_name))
    if (len(names_of_files) % 6) != 0:
        raise ValueError('The directory `{0}` contains wrong data!'.format(data_dir_name))
    list_of_all_texts = []
    list_of_all_token_bounds = []
    for idx in range(len(names_of_files) // 6):
        base_name = names_of_files[idx * 6]
        point_pos = base_name.rfind('.')
        if point_pos <= 0:
            raise ValueError('The file `{0}` has incorrect name.'.format(base_name))
        prepared_base_name = base_name[:point_pos].strip()
        if len(prepared_base_name) == 0:
            raise ValueError('The file `{0}` has incorrect name.'.format(base_name))
        tokens_file_name = os.path.join(data_dir_name, prepared_base_name + '.tokens')
        if not os.path.isfile(tokens_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(tokens_file_name))
        spans_file_name = os.path.join(data_dir_name, prepared_base_name + '.spans')
        if not os.path.isfile(spans_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(spans_file_name))
        objects_file_name = os.path.join(data_dir_name, prepared_base_name + '.objects')
        if not os.path.isfile(objects_file_name):
            raise ValueError('The file `{0}` does not exist!'.format(objects_file_name))
        list_of_texts, list_of_labels = load_document(tokens_file_name, spans_file_name, objects_file_name)
        list_of_all_texts += [list(map(lambda it: it[0], cur_text)) for cur_text in list_of_texts]
        list_of_all_token_bounds.append(
            (
                base_name,
                [list(map(lambda it: (it[1], it[2]), cur_text)) for cur_text in list_of_texts]
            )
        )
    return list_of_all_texts, list_of_all_token_bounds


def check_path_to_bert(dir_name: str) -> bool:
    if not os.path.isdir(dir_name):
        return False
    if not os.path.isfile(os.path.join(dir_name, 'vocab.txt')):
        return False
    if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.data-00000-of-00001')):
        return False
    if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.index')):
        return False
    if not os.path.isfile(os.path.join(dir_name, 'bert_model.ckpt.meta')):
        return False
    if not os.path.isfile(os.path.join(dir_name, 'bert_config.json')):
        return False
    return True


def texts_to_batch_for_bert(texts: List[List[int]], batch_size: int, max_seq_len: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    mask = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    segments = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    n = min(batch_size, len(texts))
    for text_idx in range(n):
        for token_idx in range(len(texts[text_idx])):
            tokens[text_idx][token_idx] = texts[text_idx][token_idx]
            mask[text_idx][token_idx] = 1
    if n < batch_size:
        for text_idx in range(n, batch_size):
            for token_idx in range(len(texts[n - 1])):
                tokens[text_idx][token_idx] = texts[n - 1][token_idx]
                mask[text_idx][token_idx] = 1
    return tokens, mask, segments


def texts_to_X(texts: List[List[str]], max_sentence_length: int, data_name: str, path_to_bert: str) -> np.ndarray:
    if os.path.isfile(data_name):
        with open(data_name, 'rb') as fp:
            X = pickle.load(fp)
        if not isinstance(X, np.ndarray):
            raise ValueError('The file `{0}` does not contain a `{1}` object.'.format(
                data_name, type(np.array([1, 2]))))
        if X.shape != (len(texts), max_sentence_length, EMBEDDING_SIZE):
            raise ValueError(
                'The file `{0}` contains an inadmissible `{1}` object. Shapes are wrong. Expected {2}, got {3}.'.format(
                    data_name, type(np.array([1, 2])), (len(texts), max_sentence_length, EMBEDDING_SIZE), X.shape)
            )
    else:
        path_to_bert_ = os.path.normpath(path_to_bert)
        if not check_path_to_bert(path_to_bert_):
            raise ValueError('`path_to_bert` is wrong! There are no BERT files into the directory `{0}`.'.format(
                path_to_bert))
        if os.path.basename(path_to_bert_).find('_uncased_') >= 0:
            do_lower_case = True
        else:
            if os.path.basename(path_to_bert_).find('_cased_') >= 0:
                do_lower_case = False
            else:
                do_lower_case = None
        if do_lower_case is None:
            raise ValueError('`{0}` is bad path to the BERT model, because a tokenization mode (lower case or no) '
                             'cannot be detected.'.format(path_to_bert))
        X = np.zeros((len(texts), max_sentence_length, EMBEDDING_SIZE), dtype=np.float32)
        batch_size = 4
        n_batches = int(math.ceil(len(texts) / float(batch_size)))
        max_seq_length_for_bert = 512
        with tf.Graph().as_default():
            input_ids_ = tf.placeholder(shape=(batch_size, max_seq_length_for_bert), dtype=tf.int32, name='input_ids')
            input_mask_ = tf.placeholder(shape=(batch_size, max_seq_length_for_bert), dtype=tf.int32, name='input_mask')
            segment_ids_ = tf.placeholder(shape=(batch_size, max_seq_length_for_bert), dtype=tf.int32,
                                          name='segment_ids')
            bert_config = BertConfig.from_json_file(os.path.join(path_to_bert, 'bert_config.json'))
            tokenizer = FullTokenizer(vocab_file=os.path.join(path_to_bert, 'vocab.txt'),
                                            do_lower_case=do_lower_case)
            bert_model = BertModel(config=bert_config, is_training=False, input_ids=input_ids_,
                                   input_mask=input_mask_, token_type_ids=segment_ids_,
                                   use_one_hot_embeddings=False)
            sequence_output = bert_model.sequence_output
            tvars = tf.trainable_variables()
            init_checkpoint = os.path.join(path_to_bert_, 'bert_model.ckpt')
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                tokenized_texts = []
                bert2tokens = []
                for cur_text in texts:
                    new_text = []
                    new_bert2tokens = []
                    start_pos = 0
                    for word_idx, cur_word in enumerate(cur_text):
                        bert_tokens = tokenizer.tokenize(cur_word)
                        new_text += bert_tokens
                        new_bert2tokens.append((start_pos + 1, start_pos + len(bert_tokens) + 1))
                        start_pos += len(bert_tokens)
                    if len(new_text) > (max_seq_length_for_bert - 2):
                        new_text = new_text[:(max_seq_length_for_bert - 2)]
                        new_bert2tokens = new_bert2tokens[:(max_seq_length_for_bert - 2)]
                    new_text = ['[CLS]'] + new_text + ['[SEP]']
                    tokenized_texts.append(tokenizer.convert_tokens_to_ids(new_text))
                    bert2tokens.append(tuple(new_bert2tokens))
                del tokenizer
                for batch_idx in range(n_batches):
                    start_pos = batch_idx * batch_size
                    end_pos = min(len(texts), (batch_idx + 1) * batch_size)
                    embeddings_of_texts_as_numpy = sess.run(
                        sequence_output,
                        feed_dict={
                            ph: x for ph, x in zip(
                                [input_ids_, input_mask_, segment_ids_],
                                texts_to_batch_for_bert(tokenized_texts[start_pos:end_pos], batch_size,
                                                        max_seq_length_for_bert)
                            )
                        }
                    )
                    for idx in range(end_pos - start_pos):
                        text_idx = start_pos + idx
                        for token_idx in range(min(len(texts[text_idx]), max_sentence_length)):
                            token_start, token_end = bert2tokens[text_idx][token_idx]
                            X[text_idx][token_idx] = embeddings_of_texts_as_numpy[idx][token_start:token_end].max(
                                axis=0)
                    del embeddings_of_texts_as_numpy
                for k in list(sess.graph.get_all_collection_keys()):
                    sess.graph.clear_collection(k)
        with open(data_name, mode='wb') as fp:
            pickle.dump(X, fp, protocol=2)
        tf.reset_default_graph()
    return X

def labels_to_y(labels: List[List[int]], max_sentence_length: int, data_name: str) -> np.ndarray:
    if os.path.isfile(data_name):
        with open(data_name, mode='rb') as fp:
            y = pickle.load(fp)
        if not isinstance(y, np.ndarray):
            raise ValueError('The file `{0}` does not contain a `{1}` object.'.format(
                data_name, type(np.array([1, 2]))))
        if y.shape != (len(labels), max_sentence_length, len(NAMED_ENTITIES)):
            raise ValueError(
                'The file `{0}` contains an inadmissible `{1}` object. Shapes are wrong. Expected {2}, got {3}.'.format(
                    data_name, type(np.array([1, 2])), (len(labels), max_sentence_length, len(NAMED_ENTITIES)), y.shape)
            )
    else:
        y = np.zeros((len(labels), max_sentence_length, len(NAMED_ENTITIES)), dtype=np.float32)
        for sample_idx in range(len(labels)):
            for token_idx in range(len(labels[sample_idx])):
                y[sample_idx][token_idx][labels[sample_idx][token_idx]] = 1.0
        with open(data_name, mode='wb') as fp:
            pickle.dump(y, fp, protocol=2)
    return y


def create_neural_network(max_sentence_length: int, n_classes: int, n_recurrent_units: int, dropout: float,
                          recurrent_dropout: float, use_crf: bool, l2_kernel: float=0.0,
                          l2_chain: float=0.0) -> Sequential:
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_sentence_length, EMBEDDING_SIZE)))
    model.add(
        Bidirectional(
            LSTM(n_recurrent_units, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=dropout),
            merge_mode='ave'
        )
    )
    if use_crf:
        crf = CRF(units=n_classes, learn_mode='join', test_mode='viterbi',
                  kernel_regularizer=(l2(l2_kernel) if l2_kernel > 0.0 else None),
                  chain_regularizer=(l2(l2_chain) if l2_chain > 0.0 else None))
        model.add(crf)
        model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
    else:
        if n_classes > 2:
            model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
        else:
            model.add(TimeDistributed(Dense(1, activation='logistic')))
        if n_classes > 2:
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        else:
            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def create_conditional_random_field(max_sentence_length: int, n_classes: int, l2_kernel: float,
                                    l2_chain: float) -> Sequential:
    model = Sequential()
    crf = CRF(units=n_classes, learn_mode='join', test_mode='viterbi',
              kernel_regularizer=(l2(l2_kernel) if l2_kernel > 0.0 else None),
              chain_regularizer=(l2(l2_chain) if l2_chain > 0.0 else None))
    model.add(Masking(mask_value=0.0, input_shape=(max_sentence_length, EMBEDDING_SIZE)))
    model.add(crf)
    model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
    return model

def get_temp_name() -> str:
    fp = tempfile.NamedTemporaryFile(delete=True)
    file_name = fp.name
    fp.close()
    del fp
    return file_name


def select_best_model(X: np.ndarray, y: np.ndarray, cv: List[Tuple[np.ndarray, np.ndarray]], max_sentence_length: int,
                      crf_using_mode: int, model_name: str, n_calls: int=50) -> Sequential:
    if crf_using_mode == 1:
        hyperparameter_space = [
            Real(0.0000001, 10.0, 'log-uniform', name='l2_kernel'),
            Real(0.0000001, 10.0, 'log-uniform', name='l2_chain')
        ]
    elif crf_using_mode == 2:
        hyperparameter_space = [
            Integer(8, 512, name='n_recurrent_units'),
            Real(0.0, 0.8, 'uniform', name='dropout'),
            Real(0.0, 0.8, 'uniform', name='recurrent_dropout')
        ]
    else:
        hyperparameter_space = [
            Real(0.0000001, 10.0, 'log-uniform', name='l2_kernel'),
            Real(0.0000001, 10.0, 'log-uniform', name='l2_chain'),
            Integer(8, 512, name='n_recurrent_units'),
            Real(0.0, 0.8, 'uniform', name='dropout'),
            Real(0.0, 0.8, 'uniform', name='recurrent_dropout')
        ]

    @use_named_args(hyperparameter_space)
    def objective(**params) -> float:
        f1_macro = []
        for train_index, test_index in cv:
            if crf_using_mode == 1:
                nn_model_ = create_conditional_random_field(max_sentence_length, len(NAMED_ENTITIES),
                                                            params['l2_kernel'], params['l2_chain'])
            elif crf_using_mode == 2:
                nn_model_ = create_neural_network(max_sentence_length, len(NAMED_ENTITIES), params['n_recurrent_units'],
                                                  params['dropout'], params['recurrent_dropout'], False)
            else:
                nn_model_ = create_neural_network(max_sentence_length, len(NAMED_ENTITIES), params['n_recurrent_units'],
                                                  params['dropout'], params['recurrent_dropout'],
                                                  True, params['l2_kernel'], params['l2_chain'])
            callbacks_ = [
                EarlyStopping(patience=3, verbose=0),
                ModelCheckpoint(filepath=tmp_file_name, verbose=0, save_best_only=True, save_weights_only=True)
            ]
            nn_model_.fit(X[train_index], y[train_index], batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, verbose=0,
                          callbacks=callbacks_, validation_split=VALIDATION_SPLIT)
            if os.path.isfile(tmp_file_name):
                nn_model_.load_weights(tmp_file_name)
            y_pred = nn_model_.predict(X[test_index], batch_size=BATCH_SIZE)
            y_true_ = []
            y_pred_ = []
            for sample_idx in range(len(test_index)):
                for token_idx in range(X.shape[1]):
                    if np.linalg.norm(X[test_index[sample_idx], token_idx]) <= K.epsilon():
                        break
                    y_true_.append(y[test_index[sample_idx], token_idx].argmax())
                    y_pred_.append(y_pred[sample_idx, token_idx].argmax())
            f1_macro.append(f1_score(y_true_, y_pred_, average='macro'))
            del nn_model_
            K.clear_session()
            del y_true_, y_pred_, y_pred
        f1_cv = np.array(f1_macro, dtype=np.float32).mean()
        return -f1_cv

    if os.path.isfile(model_name + '.json') and os.path.isfile(model_name + '.h5py'):
        with codecs.open(model_name + '.json', encoding='utf-8', errors='ignore', mode='r') as fp:
            json_data = fp.read()
        nn_model = model_from_json(json_data, custom_objects={'CRF': CRF})
        nn_model.load_weights(model_name + '.h5py')
        factrueval_logger.info('A neural network model has been loaded from the file `{0}`.'.format(model_name))
        return nn_model
    tmp_file_name = get_temp_name()
    try:
        best = gp_minimize(
            func=objective,
            dimensions=hyperparameter_space,
            verbose=True,
            n_calls=n_calls
        )
        if crf_using_mode == 1:
            factrueval_logger.info('Best params of the conditional random field have been selected '
                                   '(l2_kernel={0:.6f}, l2_chain={1:.6f})'.format(best.x[0], best.x[1]))
            nn_model = create_conditional_random_field(max_sentence_length, len(NAMED_ENTITIES), best.x[0], best.x[1])
        elif crf_using_mode == 2:
            factrueval_logger.info('Best params of the neural network model have been selected '
                                   '(n_recurrent_units={0}, dropout={1:.6f}, recurrent_dropout={2:.6f})'.format(
                best.x[0], best.x[1], best.x[2]))
            nn_model = create_neural_network(max_sentence_length, len(NAMED_ENTITIES), best.x[0], best.x[1], best.x[2],
                                             False)
        else:
            factrueval_logger.info('Best params of the hybrid BiLSTM-CRF model have been selected '
                                   '(n_recurrent_units={0}, dropout={1:.6f}, recurrent_dropout={2:.6f}, '
                                   'l2_kernel={3:.6f}, l2_chain={4:.6f})'.format(
                best.x[2], best.x[3], best.x[4], best.x[0], best.x[1]))
            nn_model = create_neural_network(max_sentence_length, len(NAMED_ENTITIES), best.x[2], best.x[3], best.x[4],
                                             True, best.x[0], best.x[1])
        callbacks = [
            EarlyStopping(patience=3, verbose=0),
            ModelCheckpoint(filepath=tmp_file_name, verbose=0, save_best_only=True, save_weights_only=True)
        ]
        nn_model.fit(X, y, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, verbose=1,
                     callbacks=callbacks, validation_split=VALIDATION_SPLIT)
        if os.path.isfile(tmp_file_name):
            nn_model.load_weights(tmp_file_name)
        with codecs.open(model_name + '.json', encoding='utf-8', errors='ignore', mode='w') as fp:
            fp.write('{0}'.format(nn_model.to_json()))
        nn_model.save_weights(model_name + '.h5py')
        factrueval_logger.info('A neural network model has been saved into the file `{0}`.'.format(model_name))
    finally:
        if os.path.isfile(tmp_file_name):
            os.remove(tmp_file_name)
    return nn_model


def recognized_classes_to_ne_bounds(texts_for_testing: List[List[str]],
                                    classes_distribution: np.ndarray) -> List[List[Tuple[str, Tuple[int, int]]]]:
    res = []
    for text_idx in range(len(texts_for_testing)):
        start_token_idx = -1
        ne_class = ''
        named_entities_of_text = []
        for token_idx in range(len(texts_for_testing[text_idx])):
            class_id = classes_distribution[text_idx][token_idx].argmax()
            if NAMED_ENTITIES[class_id] == 'O':
                if start_token_idx >= 0:
                    named_entities_of_text.append(
                        (
                            ne_class,
                            (start_token_idx, token_idx)
                        )
                    )
                    ne_class = ''
                    start_token_idx = -1
            else:
                if NAMED_ENTITIES[class_id].startswith('B-'):
                    if start_token_idx >= 0:
                        named_entities_of_text.append(
                            (
                                ne_class,
                                (start_token_idx, token_idx)
                            )
                        )
                    ne_class = NAMED_ENTITIES[class_id][2:]
                    start_token_idx = token_idx
                else:
                    if start_token_idx >= 0:
                        if NAMED_ENTITIES[class_id][2:] != ne_class:
                            named_entities_of_text.append(
                                (
                                    ne_class,
                                    (start_token_idx, token_idx)
                                )
                            )
                            ne_class = NAMED_ENTITIES[class_id][2:]
                            start_token_idx = token_idx
                    else:
                        ne_class = NAMED_ENTITIES[class_id][2:]
                        start_token_idx = token_idx
        if start_token_idx >= 0:
            named_entities_of_text.append(
                (
                    ne_class,
                    (start_token_idx, len(texts_for_testing[text_idx]))
                )
            )
        res.append(named_entities_of_text)
    return res


def generate_random_named_entities(texts_for_testing: List[List[str]], max_sentence_length: int) -> np.ndarray:
    types_of_named_entities = ['O', 'ORG', 'PER', 'LOC']
    y = np.zeros((len(texts_for_testing), max_sentence_length, len(NAMED_ENTITIES)), dtype=np.float32)
    for text_idx in range(len(texts_for_testing)):
        n_tokens = len(texts_for_testing[text_idx])
        token_idx = 0
        while token_idx < n_tokens:
            ne_type = random.choice(types_of_named_entities)
            ne_len = random.randint(1, 4)
            if (token_idx + ne_len) > n_tokens:
                ne_len = n_tokens - token_idx
            if ne_type == 'O':
                for idx in range(ne_len):
                    y[text_idx, token_idx, NAMED_ENTITIES.index(ne_type)] = 1.0
                    token_idx += 1
            else:
                y[text_idx, token_idx, NAMED_ENTITIES.index('B-' + ne_type)] = 1.0
                token_idx += 1
                if ne_len > 1:
                    for idx in range(ne_len - 1):
                        y[text_idx, token_idx, NAMED_ENTITIES.index('I-' + ne_type)] = 1.0
                        token_idx += 1
    return y


def test_best_model(prediction_dir_name: str, nn_model: Union[Sequential, None], max_sentence_length: int,
                    X_test: Union[np.ndarray, None], texts_for_testing: List[List[str]],
                    token_bounds_for_testing: List[Tuple[str, List[List[Tuple[int, int]]]]]):
    if (nn_model is None) or (X_test is None):
        y_test = generate_random_named_entities(texts_for_testing, max_sentence_length)
    else:
        y_test = nn_model.predict(X_test, batch_size=BATCH_SIZE)
    token_bounds_of_names_entities = recognized_classes_to_ne_bounds(texts_for_testing, y_test)
    sample_idx = 0
    for doc in token_bounds_for_testing:
        base_name = doc[0]
        point_pos = base_name.rfind('.')
        if point_pos >= 0:
            base_name = base_name[:point_pos].strip()
        if len(base_name) == 0:
            raise ValueError('The name `{0}` is wrong!'.format(base_name))
        full_name = os.path.join(prediction_dir_name, base_name + '.task1')
        with codecs.open(full_name, mode='w', encoding='utf-8', errors='ignore') as fp:
            for text_info in doc[1]:
                for cur_ne in token_bounds_of_names_entities[sample_idx]:
                    ne_class = cur_ne[0]
                    ne_token_start = cur_ne[1][0]
                    ne_token_end = cur_ne[1][1] - 1
                    ne_char_start = text_info[ne_token_start][0]
                    ne_char_end = text_info[ne_token_end][0] + text_info[ne_token_end][1]
                    fp.write('{0} {1} {2}\n'.format(ne_class.lower(), ne_char_start, ne_char_end - ne_char_start))
                sample_idx += 1


def main():
    factrueval_logger.level = logging.INFO
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the best neural network model.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='The binary file with the cached data for training and testing.')
    parser.add_argument('-r', '--res', dest='res_dir_name', type=str, required=True,
                        help='The directory into which all recognized named entity labels will be saved.')
    parser.add_argument('-t', '--type', dest='model_type', type=str, required=True,
                        choices=['crf', 'bilstm', 'bilstm-crf', 'random'],
                        help='The type of used model (CRF, BiLSTM, hybrid BiLSTM-CRF or random labeling).')
    parser.add_argument('-n', '--n_calls', dest='number_of_calls', type=int, required=False, default=50,
                        help='The total number of evaluations.')
    parser.add_argument('-b', '--bert', dest='path_to_bert', type=str, required=True, help='Path to the BERT model.')
    args = parser.parse_args()

    path_to_bert = args.path_to_bert
    result_directory_name = os.path.normpath(args.res_dir_name)
    assert os.path.isdir(result_directory_name), 'The directory `{0}` does not exist!'.format(result_directory_name)
    model_name = os.path.join(args.model_name)
    model_dir = os.path.dirname(model_name)
    if len(model_dir) > 0:
        assert os.path.isdir(model_dir), 'The directory `{0}` does not exist!'.format(model_dir)
    data_name = os.path.join(args.data_name)
    data_dir = os.path.dirname(data_name)
    if len(data_dir) > 0:
        assert os.path.isdir(data_dir), 'The directory `{0}` does not exist!'.format(data_dir)
    model_type = args.model_type
    assert model_type in POSSIBLE_MODEL_TYPES, '`{0}` is unknown model type!'.format(model_type)
    model_type = POSSIBLE_MODEL_TYPES[model_type]
    n_calls = args.number_of_calls
    assert n_calls > 10, 'The total number of evaluations must be a positive integer value greater than 10!'
    train_data_dir = os.path.join(os.path.dirname(__file__), '..', 'devset')
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'testset')
    texts_for_training, labels_for_training = load_data_for_training(train_data_dir)
    texts_for_testing, texts_and_token_bounds_for_testing = load_data_for_testing(test_data_dir)
    factrueval_logger.info('All data have been loaded...')
    max_sentence_length = max(map(lambda it: len(it), texts_for_training + texts_for_testing))
    if model_type < 1:
        test_best_model(result_directory_name, None, max_sentence_length, None, texts_for_testing,
                        texts_and_token_bounds_for_testing)
    else:
        X_train = texts_to_X(texts_for_training, max_sentence_length, data_name=(data_name + '.X_train'),
                             path_to_bert=path_to_bert)
        y_train = labels_to_y(labels_for_training, max_sentence_length, data_name=(data_name + '.y_train'))
        X_test = texts_to_X(texts_for_testing, max_sentence_length, data_name=(data_name + '.X_test'),
                            path_to_bert=path_to_bert)
        factrueval_logger.info('Data for training and testing have been prepared...')
        cv = [(train_index, test_index) for train_index, test_index in KFold(n_splits=3, shuffle=True).split(X_train)]
        best_nn_model = select_best_model(X_train, y_train, cv, max_sentence_length, model_type, model_name, n_calls)
        test_best_model(result_directory_name, best_nn_model, max_sentence_length, X_test, texts_for_testing,
                        texts_and_token_bounds_for_testing)


if __name__ == '__main__':
    main()
