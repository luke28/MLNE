from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from utils.lib_ml import MachineLearningLib as mll
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier


data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data,batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels


def word2vec(walks,vocabulary_size,params):
    # Give a folder path as an argument with '--log_dir' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    FLAGS, unparsed = parser.parse_known_args()

    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    data = []
    for item in walks:
        data = data + item
    global data_index

    batch, labels = generate_batch(data,batch_size=params['batch_size'], num_skips=2, skip_window=params['window_size'])
    #for i in range(8):
    #  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
    #        reverse_dictionary[labels[i, 0]])

    # Step 4: Build and train a skip-gram model.

    batch_size = params['batch_size']
    embedding_size = params['embedding_size']  # Dimension of the embedding vector.
    skip_window = params['window_size']  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    num_sampled = params['num_sampled']  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
          embeddings = tf.Variable(
              tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
          embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
          nce_weights = tf.Variable(
              tf.truncated_normal(
                  [vocabulary_size, embedding_size],
                  stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
          nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      # Explanation of the meaning of NCE loss:
      #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
      with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

      # Add the loss value as a scalar to summary.
      tf.summary.scalar('loss', loss)

      # Construct the SGD optimizer using a learning rate of 1.0.
      with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

      # Merge all summaries.
      merged = tf.summary.merge_all()

      # Add variable initializer.
      init = tf.global_variables_initializer()

      # Create a saver.
      saver = tf.train.Saver()

    # Step 5: Begin training.
    num_steps = 30000

    with tf.Session(graph=graph) as session:
        # Open a writer to write summaries.
        writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data,batch_size, num_skips,skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
            # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run(
                [optimizer, merged, loss],
                feed_dict=feed_dict,
                run_metadata=run_metadata)
            average_loss += loss_val

            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
            # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
              writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 2000 == 0:
              if step > 0:
                average_loss /= 2000
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step ', step, ': ', average_loss)
              average_loss = 0

        final_embeddings = normalized_embeddings.eval()
          
        return final_embeddings
 
 