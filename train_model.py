import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import csv

# Charger les embeddings GloVe (ou autres embeddings préentraînés)
def load_glove_embeddings(filepath, word_index, embedding_dim=300):
    embeddings_index = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print(f"Error parsing line for word '{word}': {line}")
                continue
            if len(coefs) == embedding_dim:
                embeddings_index[word] = coefs
            else:
                print(f"Skipping word '{word}' due to incorrect dimension ({len(coefs)} vs expected {embedding_dim})")

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Charger le dataset depuis le fichier CSV
commands = []
labels = []
with open('event_commands.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        commands.append(row['command'])
        labels.append(row['label'])

# Préparer les données
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(commands)
sequences = tokenizer.texts_to_sequences(commands)
padded_sequences = pad_sequences(sequences, padding='post')

# Encoder les labels correctement
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_sequences = np.array([label_tokenizer.word_index.get(label, 0) for label in labels])

# Charger les embeddings GloVe
embedding_dim = 100
embedding_matrix = load_glove_embeddings('glove.840B.300d.txt', tokenizer.word_index, embedding_dim)

# Définir le modèle avec embeddings préentraînés et mécanisme d'attention
input_seq = Input(shape=(padded_sequences.shape[1],))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, 
                            output_dim=embedding_dim, 
                            weights=[embedding_matrix], 
                            input_length=padded_sequences.shape[1], 
                            trainable=False)(input_seq)

lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
attention_layer = Attention()([lstm_out, lstm_out])
dropout_layer = Dropout(0.5)(attention_layer)
lstm_out_2 = LSTM(64)(dropout_layer)
dropout_layer_2 = Dropout(0.5)(lstm_out_2)

dense_layer = Dense(64, activation='relu')(dropout_layer_2)
output_layer = Dense(len(label_tokenizer.word_index) + 1, activation='softmax')(dense_layer)

model = Model(inputs=input_seq, outputs=output_layer)

# Compiler le modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, label_sequences, test_size=0.2)

# Entraîner le modèle
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Sauvegarder le modèle et les tokenizers
model.save('advanced_command_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_tokenizer.pickle', 'wb') as handle:
    pickle.dump(label_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
