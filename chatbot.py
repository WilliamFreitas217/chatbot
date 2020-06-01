import re
import time
import numpy as np
import tensorflow as tf


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()#/@;:<>{}~+=?.|,]", "", text)

    return text


talks = open("movie_conversations.txt", encoding="utf-8", errors='ignore').read().split("\n")
lines = open("movie_lines.txt", encoding="utf-8", errors='ignore').read().split("\n")

line_id = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        line_id[_line[0]] = _line[4]

talks_id = []

for talk in talks[:-1]:
    _talk = talk.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    talks_id.append(_talk.split(","))

questions = []
answers = []

for talk in talks_id:
    for i in range(len(talk) - 1):
        questions.append(line_id[talk[i]])
        answers.append(line_id[talk[i + 1]])

cleaned_questions = []
for question in questions:
    cleaned_questions.append(clean_text(question))

cleaned_answers = []
for answer in answers:
    cleaned_answers.append(clean_text(answer))

words_count = {}
for question in cleaned_questions:
    for word in question.split():
        if word not in words_count:
            words_count[word] = 1
        else:
            words_count[word] += 1

for answer in cleaned_answers:
    for word in answer.split():
        if word not in words_count:
            words_count[word] = 1
        else:
            words_count[word] += 1

limit: int = 20

questions_id = {}
word_id = 0
for word, count in words_count.items():
    if count >= limit:
        questions_id[word] = word_id
        word_id += 1

answers_id = {}
word_id = 0
for word, count in words_count.items():
    if count >= limit:
        answers_id[word] = word_id
        word_id += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questions_id[token] = len(token) + 1

for token in tokens:
    answers_id[token] = len(token) + 1

answers_words = {w_i: w for w, w_i in answers_id.items()}

for i in range(len(cleaned_answers)):
    cleaned_answers[i] += ' <EOS>'

questions_to_int = []
for question in cleaned_questions:
    ints = []
    for word in question.split():
        if word not in questions_id:
            ints.append(questions_id['<OUT>'])
        else:
            ints.append(questions_id[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in cleaned_answers:
    ints = []
    for word in answer.split():
        if word not in answers_id:
            ints.append(answers_id['<OUT>'])
        else:
            ints.append(answers_id[word])
    answers_to_int.append(ints)

cleaned_sorted_questions = []
cleaned_sorted_answers = []

for size in range(1, 26):
    for i in enumerate(questions_to_int):
        if len(i[1]) == size:
            cleaned_sorted_questions.append(questions_to_int[i[0]])
            cleaned_sorted_answers.append(answers_to_int[i[0]])


# Seq2Seq model creation
# Placeholders to entries and outputs
# Param used on the entire project


def entry_models():
    entries = tf.placeholder(tf.int32, [None, None], name="entries")
    outputs = tf.placeholder(tf.int32, [None, None], name="outputs")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return entries, outputs, learning_rate, keep_prob


# Outputs' pre-processing [Target]


def output_pre_processing(outputs, word_to_int, batch_size):
    # Filling a matrix with the Start of Sentence so the deep learning algorithm knows where it begins
    left = tf.fill([batch_size, 1], word_to_int['<SOS>'])

    # Removing the <EOS> indicator so the training would not be compromised. It's not part of the original sentence
    right = tf.strided_slice(outputs, [0, 0], [batch_size, -1], strides=[1, 1])
    outputs_pre_processed = tf.concat([left, right], 1)  # One data base next to the other
    return outputs_pre_processed


def rnn_encoder(rnn_entries, rnn_size, n_layers, keep_prob, sequence_size):
    """
    Implementing RNN encoder
    :param sequence_size: Sentence size
    :param rnn_entries: Entries
    :param rnn_size: RNN size
    :param n_layers: Number of layers on the encoder through time
    :param keep_prob: drop out value on the training set [Turn 0 the entries based on the value converted to %]
    :return:
    """
    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWraper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*n_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                       cell_bw=encoder_cell,
                                                       sequence_length=sequence_size,
                                                       inputs=rnn_entries,
                                                       dtype=tf.float32)
    return encoder_state


def training_set_decoder(encoder_state, decoder_cell, decoder_embedded_entry, sequence_size,
                         decoder_scope, output_func, keep_prob, batch_size):
    state_attention = tf.zeros([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        state_attention, attention_option='bahdanau', num_units=decoder_cell.output_size
    )
    training_decoder_func = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                          attention_keys,
                                                                          attention_values,
                                                                          attention_score_function,
                                                                          attention_construct_function,
                                                                          name='attn_dec_train')

    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, training_decoder_func,
                                                                  decoder_embedded_entry, sequence_size,
                                                                  scope=decoder_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob=keep_prob)
    return output_func(decoder_output_dropout)


def test_set_decoder(encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, max_size, n_words, sequence_size,
                         decoder_scope, output_func, keep_prob, batch_size):

    state_attention = tf.zeros([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        state_attention, attention_option='bahdanau', num_units=decoder_cell.output_size
    )
    test_decoder_func = tf.contrib.seq2seq.attention_decoder_fn_inference(output_func,
                                                                          encoder_state[0],
                                                                          attention_keys,
                                                                          attention_values,
                                                                          attention_score_function,
                                                                          attention_construct_function,
                                                                          decoder_embedded_matrix,
                                                                          sos_id, eos_id, max_size,
                                                                          n_words, name='attn_dec_inf')

    prediction_test, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_func,
                                                                   scope=decoder_scope)

    return prediction_test


def rnn_decoder(decoder_embedded_entry, decoder_embedded_matrix, encoder_state, n_words,
                sequence_size, rnn_size, n_layers, keep_prob,  batch_size):
    with tf.variable_scope("decoder") as scope_decoder:
        lstm = tf.contrib.rnn.LSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWraper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * n_layers)
        weights = tf.truncated_normal(stddev=0.1)
        biases = tf.zeros_initializer()
        output_func = lambda x: tf.contrib.layers.fully_connected(x, n_words, )