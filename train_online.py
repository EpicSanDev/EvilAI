import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# Vérifier si le fichier existe, sinon le créer
if not os.path.exists('user_interactions.log'):
    open('user_interactions.log', 'w').close()

# Charger le modèle et les tokenizers existants
if os.path.exists('advanced_command_model.h5'):
    model = load_model('advanced_command_model.h5')
    
    # Recompiler le modèle pour s'assurer que les métriques sont bien définies
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Forcer la construction des métriques en effectuant une évaluation factice
    dummy_data = np.zeros((1, model.input_shape[1]))
    dummy_label = np.zeros((1,))
    model.evaluate(dummy_data, dummy_label, verbose=0)
else:
    raise FileNotFoundError("Le fichier command_model.h5 est introuvable. Veuillez entraîner le modèle et le sauvegarder avant de continuer.")

# Charger les tokenizers
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_tokenizer.pickle', 'rb') as handle:
    label_tokenizer = pickle.load(handle)

# Charger les nouvelles données d'interaction utilisateur
new_data = []
with open('user_interactions.log', 'r') as file:
    for line in file:
        command, action = line.strip().split(',')
        new_data.append((command, action))

# Réentraîner le modèle uniquement s'il y a des nouvelles données
if new_data:
    commands, labels = zip(*new_data)
    sequences = tokenizer.texts_to_sequences(commands)
    padded_sequences = pad_sequences(sequences, padding='post')
    
    # Convertir chaque étiquette en un entier unique
    label_sequences = np.array([label_tokenizer.texts_to_sequences([label])[0][0] for label in labels])

    # Re-entraîner le modèle sur les nouvelles données
    model.fit(padded_sequences, label_sequences, epochs=5)

    # Sauvegarder le modèle mis à jour
    model.save('advanced_command_model.h5')

    # Vider le fichier de log après réentraînement
    open('user_interactions.log', 'w').close()
