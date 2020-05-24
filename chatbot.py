
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

cleaned_conversations = []
for question in questions:
    cleaned_conversations.append(clean_text(question))

cleaned_conversations = []
for answer in answers:
    cleaned_conversations.append(clean_text(answer))
