embedding_dims = 300
MAX_FEATURES =72039
MAX_TEXT_LENGTH = 400
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

GLOVE_EMBEDDING_FILE=path+'glove.840B.300d.txt'

embedding_path=path+'crawl-300d-2M.vec'

stop_words = {'the', 'a', 'an'}
stop_words.update(
    ['.', ',', '"', "'", '?', ':', ';', '(', ')', '[', ']', '{', '}','\'\'','``','...','-','%'])  # remove it if you need punctuation

