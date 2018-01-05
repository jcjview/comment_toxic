# encoding=utf-8
import math
from collections import defaultdict
import nltk
import pandas as pd
from nltk.tokenize import TweetTokenizer
import config

porter = nltk.PorterStemmer()

train = pd.read_csv('data/train.csv')
toxic_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
count_word = defaultdict(lambda: 0)
count_class = defaultdict(lambda: 0)
count_word_class = defaultdict(lambda: defaultdict(int))
badwords = config.badwords

print(len(badwords))
fptemp = open("pmi_temp_badwords.txt", 'w', encoding="utf-8")
N = 0
all_word = 0
all_class =  defaultdict(lambda: 0)
for index, row in train.iterrows():
    line = row['comment_text']
    for label in toxic_list:
        all_class[label] += 1
        istrue = row[label]
        if istrue == 1:
            count_class[label] = count_class[label] + 1
    tokens = nltk.word_tokenize(line)
    for t in tokens:
        all_word += 1
        word=t
        # word = porter.stem(t)
        word = word.lower()
        word = word.replace('\t', '')
        if word in config.stop_words:
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

comments = pd.read_csv('./baseline/toxicity_comments.tsv', sep = '\t', index_col = 0)

annotations = pd.read_csv('./baseline/toxicity_annotations.tsv',  sep = '\t')
toxicity_labels = annotations.groupby('rev_id')['toxicity'].mean()>0.5

result = pd.concat([comments,annotations], axis=1)
label='toxic'
for index, row in result.iterrows():
    line = row['comment_text']
    istrue = row['toxicity']
    all_class[label]+= 1
    if istrue:
        count_class[label] = count_class[label] + 1
        tokens = nltk.word_tokenize(line)
        for t in tokens:
            all_word += 1
            word = word.lower()
            word = word.replace('\t', '')
            if word in config.stop_words:
                continue
            if word in badwords:
                fptemp.write(word + "=>" + badwords[word] + "\n")
                word = badwords[word] + "_badwords"
            count_word[word] += 1
            for label in toxic_list:
                istrue = row[label]
                if istrue == 1:
                    count_word_class[label][word] += 1
                    N += 1


fp2 = open("pmi_save2.csv", 'w', encoding="utf-8")
with open("pmi_save1.csv", 'w', encoding="utf-8") as fp:
    for word in count_word:
        cw = 1.0 * count_word[word]#tf
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
            p_class = cl / all_class[label]
            p_word = cw / all_word
            if p_class == 0: p_class = 1
            if p_word == 0: p_word = 1
            pmi = cwc / N / p_class / p_word
            if pmi > 0:
                # print(cwc)
                logpmi = math.log2(pmi)
            else:
                logpmi = 0
            if logpmi <0:
                logpmi=0
            fp.write(str(logpmi))
            fp.write('\t')

            fp2.write(str(cwc))
            fp2.write('\t')
        # fp2.write(str(N))
        # fp2.write('\t')
        # fp2.write(str(cl))
        # fp2.write('\t')
        # fp2.write(str(all_class[label]))
        # fp2.write('\t')
        fp2.write(str(cw))
        fp2.write('\t')
        fp2.write(str(all_word))
        fp2.write('\t\t')

        fp.write(str(cw))
        fp.write('\t')
        fp2.write('\n')
        fp.write('\n')
fp2.close()
