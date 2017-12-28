# encoding=utf-8
import math
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import pandas as pd

porter = nltk.PorterStemmer()


toxic_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
count_word = defaultdict(lambda: 0)
count_class = defaultdict(lambda: 0)
count_word_class = defaultdict(lambda: defaultdict(int))
badwords = {}
stop_words = set(stopwords.words('english'))
stop_words.update(
    ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','\'\'','``','...','-','%'])  # remove it if you need punctuation

with open("data/badwords.txt") as fp:
    for line in fp:
        line = line.strip()
        lines = line.split(',')
        if len(lines) == 1:
            badwords[line] = line
        elif len(lines) == 2:
            badwords[lines[0]] = line[1]
print(len(badwords))
N = 0
all_word = 0
all_class = 0
outf = open("./1.txt",'w',encoding='utf-8')
# train = pd.read_csv('data/train.csv')
# for index, row in train.iterrows():
#     line = row['comment_text']
#     tokens = nltk.word_tokenize(line)
#     for t in tokens:
#         word = porter.stem(t)
#         word = word.lower()
#         word = word.replace('\t', '')
#         if word in stop_words:
#             continue
#         if word in badwords:
#             word = badwords[word]
#         outf.write(word)
#         outf.write(' ')
#     outf.write("\n")

train = pd.read_csv("./data/test.csv")
train=train["comment_text"].fillna("MISSINGVALUE").values
for line in train:
    tokens = nltk.word_tokenize(line)
    for t in tokens:
        word = porter.stem(t)
        word = word.lower()
        word = word.replace('\t', '')
        if word in stop_words:
            continue
        if word in badwords:
            word = badwords[word]
        outf.write(word)
        outf.write(' ')
    outf.write("\n")
outf.close()