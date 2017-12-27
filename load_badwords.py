# encoding=utf-8
import math
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import pandas as pd

porter = nltk.PorterStemmer()

train = pd.read_csv('data/train.csv')
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
fptemp = open("pmi_temp_badwords.txt", 'w', encoding="utf-8")
print(len(badwords))
N = 0
all_word = 0
all_class = 0
for index, row in train.iterrows():
    line = row['comment_text']
    for label in toxic_list:
        istrue = row[label]
        if istrue == 1:
            count_class[label] = count_class[label] + 1
            all_class += 1
    tokens = nltk.word_tokenize(line)
    for t in tokens:
        all_word += 1
        word = porter.stem(t)
        word = word.lower()
        word = word.replace('\t', '')
        if word in stop_words:
            continue
        if word in badwords:
            fptemp.write(word+"=>"+badwords[word]+"\n")
            word = badwords[word] + "_badwords"
        count_word[word] += 1
        for label in toxic_list:
            istrue = row[label]
            if istrue == 1:
                count_word_class[label][word] += 1
                N += 1
badwords = ""
train = ""
fptemp.close()
# data_save.append(['word','toxic','severe_toxic','obscene','threat','insult','identity_hate'])
print(N)

fp2 = open("pmi_save2.csv", 'w', encoding="utf-8")
with open("pmi_save1.csv", 'w', encoding="utf-8") as fp:
    for word in count_word:
        cw = 1.0 * count_word[word]
        if cw == 0:
            cw = 1
        fp.write(word)
        fp.write('\t')
        fp2.write(word)
        fp2.write('\t')
        for label in toxic_list:
            cl = 1.0 * count_class[label]
            cwc = 1.0 * count_word_class[label][word]
            if cl == 0:
                cl = 1
            p_class = cl / all_class
            p_word = cw / all_word
            if p_class == 0: p_class = 1
            if p_word == 0: p_word = 1
            pmi = cwc / N / p_class / p_word
            if pmi > 0:
                # print(cwc)
                logpmi = math.log2(pmi)
            else:
                logpmi = 0
            fp.write(str(logpmi))
            fp.write('\t')

            fp2.write(str(cwc))
            fp2.write('\t')
            fp2.write(str(N))
            fp2.write('\t')
            fp2.write(str(cl))
            fp2.write('\t')
            fp2.write(str(all_class))
            fp2.write('\t')
            fp2.write(str(cw))
            fp2.write('\t')
            fp2.write(str(all_word))
            fp2.write('\t|\t')
        fp2.write('\n')
        fp.write('\n')
fp2.close()
