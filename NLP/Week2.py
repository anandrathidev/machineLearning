# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:41:22 2018

@author: anandrathi
"""

import os
os.getcwd()

npath = "D:/Users/anandrathi/DataScience/Coursera/NLP/natural-language-processing-master\week2"
os.chdir(npath )
os.getcwd()

#In[]
import sys
sys.path.append("..")
from common.download_utils import download_week2_resources

download_week2_resources()

# In[]:
def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token

            ######################################
            ######### YOUR CODE HERE #############
            ######################################
            if token.strip().lower().startswith("http://") or token.strip().lower().startswith("https://")  or bool(urlparse(token.strip().lower()).netloc):
:              token = "<URL>"
            if token.strip().lower().startswith("@") :
               token = "<USR>"
            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags

# In[]:
train_tokens, train_tags = read_data('data/train.txt')
validation_tokens, validation_tags = read_data('data/validation.txt')
test_tokens, test_tags = read_data('data/test.txt')

# In[]:
for i in range(3):
    for token, tag in zip(train_tokens[i], train_tags[i]):
        print('%s\t%s' % (token, tag))
    print()

 # In[]:
from collections import defaultdict

 # In[]:

def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []

    # Create mappings from tokens (or tags) to indices and vice versa.
    # At first, add special tokens (or tags) to the dictionaries.
    # The first special token must have index 0.

    # Mapping tok2idx should contain each token or tag only once.
    # To do so, you should:
    # 1. extract unique tokens/tags from the tokens_or_tags variable, which is not
    #    occur in special_tokens (because they could have non-empty intersection)
    # 2. index them (for example, you can add them into the list idx2tok
    # 3. for each token/tag save the index into tok2idx).

    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    #Extract Unique tokens
    flat_list = [item for sublist in tokens_or_tags  for item in sublist]
    flat_list = list(set(flat_list))
    #intersection_tok = list(set(tokens_or_tags).intersection(special_tokens))
    Uncommon = set(flat_list)  - set(special_tokens)

    counter = 0
    for token in special_tokens:
       tok2idx[token] = counter
       idx2tok.append(token)
       counter = counter + 1
    for token in Uncommon:
       tok2idx[token] = counter
       idx2tok.append(token)
       counter = counter + 1


    return tok2idx, idx2tok

 # In[]:
special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

# Create dictionaries
token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)

# In[]:

 def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]


# In[]:

def batches_generator(batch_size, tokens, tags,
                      shuffle=True, allow_smaller_last_batch=True):
    """Generates padded batches of tokens and tags."""

    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))

        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths


 # In[]:
import tensorflow as tf
import numpy as np

 # In[]:
class BiLSTMModel():
    pass

 # In[]:
def declare_placeholders(self):
    """Specifies placeholders for the model."""

    # Placeholders for input and ground truth output.
    self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch')
    ######### YOUR CODE HERE #############
    self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')

    # Placeholder for lengths of the sequences.
    self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')

    # Placeholder for a dropout keep probability. If we don't feed
    # a value for this placeholder, it will be equal to 1.0.
    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])

    # Placeholder for a learning rate (tf.float32).
    ######### YOUR CODE HERE #############
    self.learning_rate_ph = tf.placeholder_with_default(tf.cast(0.5, tf.float32), shape=[])

# In[]:
BiLSTMModel.__declare_placeholders = classmethod(declare_placeholders)

 # In[]:
 def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
    """Specifies bi-LSTM architecture and computes logits for inputs."""

    # Create embedding variable (tf.Variable) with dtype tf.float32
    initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
    ######### YOUR CODE HERE #############
    embedding_matrix_variable =

    # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units
    # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
    forward_cell =  ######### YOUR CODE HERE #############
    backward_cell =  ######### YOUR CODE HERE #############

    # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
    # Shape: [batch_size, sequence_len, embedding_dim].
    embeddings =  ######### YOUR CODE HERE #############

    # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
    # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn].
    # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
    (rnn_output_fw, rnn_output_bw), _ =  ######### YOUR CODE HERE #############
    rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

    # Dense layer on top.
    # Shape: [batch_size, sequence_len, n_tags].
    self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)



  # In[]:

