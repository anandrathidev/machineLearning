# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:41:22 2018
@author: anandrathi
"""

import os
os.getcwd()
from urllib.parse import urlparse

npath = "D:/Users/anandrathi/DataScience/Coursera/NLP/natural-language-processing-master\week2"
npath = "C:/Users/anarathi/Documents/Coursera/natural-language-processing-master/week2/"
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
              token = "<URL>"
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

train_tokens_list = list(set([t for tl in train_tokens for t in tl]))
print(len(train_tokens_list))
train_tags_list = list(set([t for tl in train_tags for t in tl]))
print(len(train_tags_list))


test_tokens_list = list(set([t for tl in test_tokens for t in tl]))
print(len(test_tokens_list))
test_tags_list = list(set([t for tl in test_tags for t in tl]))
print(len(test_tags_list))


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


def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
    """Specifies bi-LSTM architecture and computes logits for inputs."""

    # Create embedding variable (tf.Variable) with dtype tf.float32
    initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
    ######### YOUR CODE HERE #############
    embedding_matrix_variable = tf.Variable(initial_embedding_matrix, name='embeddings_matrix', trainable = True, use_resource = True, dtype=tf.float32)

    # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units
    # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
    
    forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_rnn, state_is_tuple=True)   ######### YOUR CODE HERE #############
    forward_cell = tf.nn.rnn_cell.DropoutWrapper(forward_cell, 
                                                 output_keep_prob=self.dropout_ph)
    
    backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_rnn, state_is_tuple=True) ######### YOUR CODE HERE #############
    forward_cell = tf.nn.rnn_cell.DropoutWrapper(backward_cell, 
                                                 input_keep_prob=self.dropout_ph, 
                                                 output_keep_prob=self.dropout_ph)

    # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).    
    # Shape: [batch_size, sequence_len, embedding_dim].
    #embeddings =  tf.nn.embedding_lookup ######### YOUR CODE HERE #############
    embeddings = tf.nn.embedding_lookup(params = embedding_matrix_variable, ids = self.input_batch)
    # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
    # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn].
    # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
    # (rnn_output_fw, rnn_output_bw), _ =  ######### YOUR CODE HERE #############
    
    (rnn_output_fw, rnn_output_bw), _ =  tf.nn.bidirectional_dynamic_rnn(
        cell_fw=forward_cell, cell_bw=backward_cell,
        dtype=tf.float32,
        inputs=embeddings,
        sequence_length=self.lengths
    )
    
    rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

    # Dense layer on top.
    # Shape: [batch_size, sequence_len, n_tags].
    self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)



def compute_predictions(self):
    """Transforms logits to probabilities and finds the most probable tags."""
    
    # Create softmax (tf.nn.softmax) function
    softmax_output = tf.nn.softmax(self.logits)     ######### YOUR CODE HERE #############
    # Use argmax (tf.argmax) to get the most probable tags
    # Don't forget to set axis=-1
    # otherwise argmax will be calculated in a wrong way
    self.predictions = tf.argmax(softmax_output, axis = -1) ######### YOUR CODE HERE #############  


def compute_loss(self, n_tags, PAD_index):
    """Computes masked cross-entopy loss with logits."""
    
    # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits)
    ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
    #loss_tensor =  ######### YOUR CODE HERE #############
    loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_tags_one_hot, logits=self.logits)
    mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), tf.float32)
    # Create loss function which doesn't operate with <PAD> tokens (tf.reduce_mean)
    # Be careful that the argument of tf.reduce_mean should be
    # multiplication of mask and loss_tensor.
    self.loss =   tf.reduce_mean(tf.reduce_sum(mask * loss_tensor, axis = -1)/tf.reduce_sum(mask,axis = -1)) ######### YOUR CODE HERE #############  
    
def perform_optimization(self):
    """Specifies the optimizer and train_op for the model."""
    
    # Create an optimizer (tf.train.AdamOptimizer)
    self.optimizer =  tf.train.AdamOptimizer(self.learning_rate_ph) ######### YOUR CODE HERE #############
    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
    
    # Gradient clipping (tf.clip_by_norm) for self.grads_and_vars
    # Pay attention that you need to apply this operation only for gradients 
    # because self.grads_and_vars also contains variables.
    # list comprehension might be useful in this case.
    clip_norm = tf.cast(1.0, tf.float32)
    self.grads_and_vars =  [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars] ######### YOUR CODE HERE #############
    
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)    


def init_model(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
    self.__declare_placeholders()
    self.__build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
    self.__compute_predictions()
    self.__compute_loss(n_tags, PAD_index)
    self.__perform_optimization()  
    

  
def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
      feed_dict = {self.input_batch: x_batch,
                   self.ground_truth_tags: y_batch,
                   self.learning_rate_ph: learning_rate,
                   self.dropout_ph: dropout_keep_probability,
                   self.lengths: lengths}
      
      session.run(self.train_op, feed_dict=feed_dict)
  

def predict_for_batch(self, session, x_batch, lengths):
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    feed_dict = {self.input_batch: x_batch, 
                 self.lengths: lengths
                }
    predictions = session.run(self.predictions, feed_dict = feed_dict)
    return predictions
  
BiLSTMModel.__declare_placeholders = classmethod(declare_placeholders)
BiLSTMModel.__build_layers = classmethod(build_layers)
BiLSTMModel.__compute_predictions = classmethod(compute_predictions)
BiLSTMModel.__compute_loss = classmethod(compute_loss)
BiLSTMModel.__perform_optimization = classmethod(perform_optimization)
BiLSTMModel.__init__ = classmethod(init_model)
BiLSTMModel.train_on_batch = classmethod(train_on_batch)
BiLSTMModel.predict_for_batch = classmethod(predict_for_batch)
  
  
# In[]:
from evaluation import precision_recall_f1

def predict_tags(model, session, token_idxs_batch, lengths):
    """Performs predictions and transforms indices to tokens and tags."""
    
    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)
    
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch
    
    
def eval_conll(model, session, tokens, tags, short_report=True):
    """Computes NER quality measures using CONLL shared task script."""
    
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]): 
            if token != '<PAD>':
                ground_truth_tags.append(idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)

        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])
        
    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
    return results
    
tf.reset_default_graph()

model =  BiLSTMModel(20526, 21, 200, 200, token2idx['<PAD>'])######### YOUR CODE HERE #############


batch_size = 32*2 ######### YOUR CODE HERE #############
n_epochs = 30 ######### YOUR CODE HERE #############
learning_rate = 0.09 ######### YOUR CODE HERE #############
learning_rate_decay = 1.3 ######### YOUR CODE HERE #############
dropout_keep_probability = 0.25 ######### YOUR CODE HERE #############

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, train_tokens, train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)

    # Train the model
    for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)

    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay

print('...training finished.')
