embedding_dims = 300
MAX_FEATURES = 72039
MAX_TEXT_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 5
VALIDATION_SPLIT = 0.1
CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


train_token_path='./train_corps.txt'
test_token_path='./test_corps.txt'