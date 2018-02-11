embedding_dims = 300
MAX_FEATURES =72039
MAX_TEXT_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.1
SPLIT=10000
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
path = '../data/'

TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'
train_token_path='./train_corps.txt'
test_token_path='./test_corps.txt'
TRAIN_VALID_FILE=path+'train_valid_test.csv'

GLOVE_EMBEDDING_FILE='d:\\Users\\xj\\Downloads\\glove-840b-tokens-300d-vectors\\glove.840B.300d.txt'

stop_words = {'the', 'a', 'an'}
stop_words.update(
    ['.', ',', '"', "'", '?', ':', ';', '(', ')', '[', ']', '{', '}','\'\'','``','...','-','%'])  # remove it if you need punctuation

badwords = {}

with open("../data/badwords.txt",encoding='utf-8') as fp:
    for line in fp:
        line = line.lower().strip()
        lines = line.split(',')
        if len(lines) == 1 and line not in badwords:
            badwords[line] = line
        elif len(lines) == 2 and lines[0] not in badwords:
            badwords[lines[0].strip()] = lines[1].strip().replace(" ", "_")

with open("../data/stopwords.txt",encoding='utf-8') as fp:
    for line in fp:
        stop_words.add(line.strip())
