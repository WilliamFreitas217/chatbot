
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
    for i in range(len(talk)-1):
        questions.append(line_id[talk[i]])
        answers.append(line_id[talk[i+1]])

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
    questions_id[token] = len(token)+1

for token in tokens:
    answers_id[token] = len(token)+1

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
