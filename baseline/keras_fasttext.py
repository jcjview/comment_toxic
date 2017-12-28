import pandas as pd
import numpy as np


from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
X_train = train_df["comment_text"].fillna("sterby").values
y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test_df["comment_text"].fillna("sterby").values
# Set parameters:
max_features = 20000
maxlen = 65
batch_size = 32
embedding_dims = 200
epochs = 50
VALIDATION_SPLIT = 0.05

print('Tokenizing data...')
tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))
x_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(X_test)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')
comment_input = Input((maxlen,))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
comment_emb = Embedding(max_features, embedding_dims, input_length=maxlen)(comment_input)

# we add a GlobalMaxPool1D, which will extract information from the embeddings
# of all words in the document
main = GlobalMaxPool1D()(comment_emb)

# We project onto a six-unit output layer, and squash it with sigmoids:
output = Dense(6, activation='sigmoid')(main)

model = Model(inputs=comment_input, outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
hist = model.fit(x_train, y_train,shuffle=True, batch_size=batch_size, epochs=epochs, validation_split=VALIDATION_SPLIT)
y_pred = model.predict(x_test)
submission = pd.read_csv("./data/sample_submission.csv")
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv("submission_fasttext_2.csv", index=False)