from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, current_user, logout_user, login_required
from app import db, login_manager
from app.models import User, Event
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Créer un Blueprint nommé 'main'
main_routes = Blueprint('main', __name__)

# Charger le modèle et les tokenizers au démarrage de l'application
model = tf.keras.models.load_model('advanced_command_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_tokenizer.pickle', 'rb') as handle:
    label_tokenizer = pickle.load(handle)

# Fonction pour charger un utilisateur par son ID
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Route pour la page d'accueil
@main_routes.route('/')
def index_view():
    return render_template('index.html')

# Route pour la connexion
@main_routes.route('/login', methods=['GET', 'POST'])
def login_view():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('main.dashboard_view'))
        else:
            message = 'Login unsuccessful. Please check username and password.'
            return render_template('login.html', message=message)
    return render_template('login.html')

# Route pour le tableau de bord
@main_routes.route('/dashboard')
@login_required
def dashboard_view():
    events = Event.query.filter_by(author=current_user).all()
    return render_template('dashboard.html', events=events)

# Route pour la déconnexion
@main_routes.route('/logout')
def logout_view():
    logout_user()
    return redirect(url_for('main.index_view'))


@main_routes.route('/signup', methods=['GET', 'POST'])
def signup_view():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Vérifiez si l'utilisateur existe déjà
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('main.signup_view'))
        
        # Créez un nouvel utilisateur
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.')
        return redirect(url_for('main.login_view'))
    
    return render_template('signup.html')

@main_routes.route('/get_events')
@login_required
def get_events():
    events = Event.query.filter_by(author=current_user).all()
    events_list = []
    for event in events:
        events_list.append({
            'title': event.title,
            'start': event.date.strftime("%Y-%m-%dT%H:%M:%S"),  # Format adapté pour FullCalendar
            'allDay': False
        })
    return jsonify(events_list)

@main_routes.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    # Convertir la commande utilisateur en séquence de tokens
    sequences = tokenizer.texts_to_sequences([user_message])
    padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')
    
    # Utiliser le modèle pour prédire l'action
    prediction = model.predict(padded_sequences)
    label_index = np.argmax(prediction)
    
    # Trouver l'action correspondante
    action = None
    for label, index in label_tokenizer.word_index.items():
        if index == label_index:
            action = label
            break

    response = ""

    # Logique basée sur l'action prédite
    if action == "add_event":
        try:
            parts = user_message.split("le")
            title = parts[0].replace("ajouter événement", "").strip()
            datetime_str = parts[1].strip()
            event_date = datetime.strptime(datetime_str, "%Y-%m-%d à %H:%M")

            # Créer et ajouter l'événement
            new_event = Event(title=title, date=event_date, author=current_user)
            db.session.add(new_event)
            db.session.commit()
            response = f"Événement '{title}' ajouté le {event_date.strftime('%d %B %Y à %H:%M')}."
        except Exception as e:
            response = "Désolé, je n'ai pas compris la date ou l'heure. Veuillez fournir dans le format 'AAAA-MM-JJ à HH:MM'."
    
    else:
        response = "Je ne sais pas comment gérer cela. Essayez de dire 'ajouter événement Réunion le 2024-09-01 à 14:00'."

    return jsonify({'response': response})


# Route pour ajouter un événement
@main_routes.route('/addevent', methods=['POST'])
@login_required
def addevent_view():
    title = request.form.get('title')
    event_date = request.form.get('date')
    event = Event(title=title, date=event_date, author=current_user)
    db.session.add(event)
    db.session.commit()
    return redirect(url_for('main.dashboard_view'))

# Fonction pour prédire l'action en fonction d'une commande utilisateur
def predict_action(command):
    sequences = tokenizer.texts_to_sequences([command])
    padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')
    prediction = model.predict(padded_sequences)
    label_index = np.argmax(prediction)
    for label, index in label_tokenizer.word_index.items():
        if index == label_index:
            return label
    return None

# Route pour gérer les commandes vocales
@main_routes.route('/voice-command', methods=['POST'])
def voice_command_view():
    command = request.form.get('command')
    action = predict_action(command)
    
    if action:
        log_user_interaction(command, action)
        return jsonify({"action": action}), 200
    
    return "Action non reconnue", 400

# Fonction pour enregistrer les interactions utilisateur pour l'apprentissage
def log_user_interaction(command, action):
    with open('user_interactions.log', 'a') as file:
        file.write(f"{command},{action}\n")
