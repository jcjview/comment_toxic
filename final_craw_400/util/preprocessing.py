import pickle
import re

import h5py
import multiprocessing
import numpy as np
# from keras import initializations
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from config import *
from tqdm import tqdm
import mmap


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

TRAIN_HDF5 = './train_hdf5.h5'
word_index_path='./word_index.pkl'

wiki_reg=r'https?://en.wikipedia.org/[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
url_reg=r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
ip_reg='\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
WIKI_LINK=' WIKILINKREPLACER '
URL_LINK=' URLLINKREPLACER '
IP_LINK=' IPLINKREPLACER '

bad_dict = {}
with open("../data/badwords.list",encoding='utf-8') as fp:
    lines = fp.readlines()
    lines = [l.lower().strip() for l in lines]
    lines = [l.split(',') for l in lines]
    for v in lines:
        if len(v) == 2:
            bad_dict[v[0]] = v[1]

print("bad_dict",len(bad_dict))
from rake_nltk import Rake
def rake_parse(line):
    r = Rake()
    r.extract_keywords_from_text(line)
    word_combines = r.get_ranked_phrases()
    word_combines = [k for k in word_combines if len(k.split()) > 1]
    # filter out bad word combines
    bad_word_dict = bad_dict
    word_replacer = {}
    for k in word_combines:
        if any(map(lambda x : k.find(x) >= 0, bad_word_dict.values())):
            continue
        word_replacer[k] = '_'.join(k.split())

    for k,v in word_replacer.items():
        line = line.replace(k,v)
    return line

def dump_X_Y_train_test(train, test, y,word_index):
    print('save to h5 ',TRAIN_HDF5)
    outh5file = h5py.File(TRAIN_HDF5, 'w')
    test_token = outh5file.create_dataset('test_token', data=test)
    train_token = outh5file.create_dataset('train_token', data=train)
    train_label = outh5file.create_dataset(name='train_label', data=y)
    outh5file.close()
    with open(word_index_path, 'wb') as f:
        pickle.dump(word_index, f)

def load_train_test_y():
    print('load from h5 ',TRAIN_HDF5)
    outh5file = h5py.File(TRAIN_HDF5, 'r')
    train = outh5file['train_token']
    test = outh5file['test_token']
    y = outh5file['train_label']
    word_dict={}
    with open(word_index_path, 'rb') as f:
        word_dict= pickle.load(f)
    return train, test, y,word_dict

def worker(index,comments):
    for i,text in enumerate (comments):
        comments[i]=text_to_wordlist(text)
    print("{} done",index)
def multi_preprocess(job=2,comments=[]):
    batch=len(comments)//job+1
    pool_size = multiprocessing.cpu_count()+1
    print("pool_size", pool_size)
    pool = multiprocessing .Pool(pool_size)
    # resultList = pool.map(worker, comments)
    pool_outputs = pool.map(text_to_wordlist, comments)
    # for i in range(job):
    #     print(i,i*batch,(i+1)*batch,len(comments))
    #     if i*batch>=len(comments):
    #         break
    #     elif (i+1) * batch >= len(comments):
    #         p = pool.apply_async(func=worker, args=(i,comments[i*batch:len(comments)],))
    #         # p.start()
    #         # p.join()
    #     else:
    #         p = pool.apply_async(func=worker, args=(i,comments[i * batch: (i+1) * batch],))
    #         # p.start()
    #         # p.join()
    #     print("start", i)
    pool.close()
    pool.join()
    print('successful')
    return pool_outputs
def get_X_train_X_test(train_df, test_df):
    print('Processing text dataset')
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values
    print('Processing new')
    import copy

    comments = list(list_sentences_train)
    comments=multi_preprocess(job=4,comments=comments)
    # for text in tqdm(list_sentences_train):
    #     comments.append(text_to_wordlist(text))
    test_comments = list(list_sentences_test)
    test_comments=multi_preprocess(job=4,comments=test_comments)
    # for text in tqdm(list_sentences_test):
    #     test_comments.append(text_to_wordlist(text))

    with open('text.txt','w',encoding='utf-8') as fp:
         # for c in comments:
        fp.write("\n".join(comments))
        fp.write("\n".join(test_comments))
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_TEXT_LENGTH)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)

    test_data = pad_sequences(test_sequences, maxlen=MAX_TEXT_LENGTH)
    print('Shape of test_data tensor:', test_data.shape)


    return data, test_data, tokenizer.word_index

# Regex to remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^A-Za-z_\d?! ]', re.IGNORECASE)
# regex to replace all numerics
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

import string
def character_range(text):
    for ch in string.ascii_lowercase[:27]:
        if ch in text:
            template=r"("+ch+")\\1{3,}"
            text = re.sub(template, ch, text)
    return text


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    #clear link
    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(url_reg, text)
    for u in c:
        text = text.replace(u, WIKI_LINK)
    c = re.findall(wiki_reg, text)
    for u in c:
        text = text.replace(u, URL_LINK)
    c = re.findall(ip_reg, text)
    for u in c:
        text = text.replace(u, IP_LINK)

    text=character_range(text)
    bad_word_dict = bad_dict
    # Regex to remove all Non-Alpha Numeric and space
    special_character_removal = re.compile(r'[^A-Za-z\d!?*\'.,; ]', re.IGNORECASE)
    # regex to replace all numerics
    replace_numbers = re.compile(r'\b\d+\b', re.IGNORECASE)
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Remove Special Characters
    text = special_character_removal.sub(' ', text)
    for k, v in bad_word_dict.items():
        # bad_reg = re.compile('[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ]'+ re.escape(k) +'[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n ]')
        bad_reg = re.compile('[\W]?' + re.escape(k) + '[\W]|[\W]' + re.escape(k) + '[\W]?')
        text = bad_reg.sub(' ' + v + ' ', text)
        '''
        bad_reg = re.compile('[\W]'+ re.escape(k) +'[\W]?')
        text = bad_reg.sub(' '+ v, text)
        bad_reg = re.compile('[\W]?'+ re.escape(k) +'[\W]')
        text = bad_reg.sub(v + ' ', text)
        '''

    # Replace Numbers
    text = replace_numbers.sub('NUMBERREPLACER', text)
    text = text.split()
    text = " ".join(text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    # rake parsing
    text = rake_parse(text)
    return text


"""
prepare embeddings
"""


def get_embedding_matrix(word_index,Emed_path):
    #    Glove Vectors
    print('Indexing word vectors')
    embeddings_index = {}
    file_line=get_num_lines(Emed_path)
    print('lines ',file_line)
    with open(Emed_path,encoding='utf-8') as f:
        for line in tqdm(f, total=file_line):
            values = line.split()
            word = ' '.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            embeddings_index[word] = coefs
    # f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = min(MAX_FEATURES, len(word_index))
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(loc=emb_mean, scale=emb_std, size=(nb_words, embedding_dims))
    # embedding_matrix = np.zeros((nb_words, embedding_dims))
    word_in_embeded=0
    for word, i in tqdm(word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            word_in_embeded+=1
    # np.save(Emed_path, embedding_matrix)
    word_in_embeded=nb_words-word_in_embeded
    print('Null word embeddings: %d' % word_in_embeded)
    return embedding_matrix

