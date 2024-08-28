import random
import csv
from datetime import datetime, timedelta

# Listes de phrases pour chaque intention
add_event_phrases = [
    "ajouter un événement {event} le {date} à {time}",
    "ajoute un événement {event} pour le {date} à {time}",
    "prévoir {event} le {date} à {time}",
    "calendrier {event} le {date} à {time}",
    "planifier {event} le {date} à {time}"
]

delete_event_phrases = [
    "supprimer l'événement {event} du {date}",
    "supprime l'événement {event} du {date}",
    "annuler {event} prévu le {date}",
    "retire {event} du calendrier le {date}",
    "efface l'événement {event} du {date}"
]

update_event_phrases = [
    "modifie l'événement {event} à {time}",
    "change l'heure de l'événement {event} à {time}",
    "replanifier {event} pour {time}",
    "déplace l'événement {event} à {time}",
    "met à jour l'événement {event} pour {time}"
]

get_event_phrases = [
    "quand est ma {event} ?",
    "montre-moi les événements du {date}",
    "quand ai-je prévu {event} ?",
    "rappelle-moi quand est {event}",
    "quels événements sont prévus le {date} ?"
]

events = ["réunion", "dîner", "conférence", "appel", "entretien", "cours", "formation"]
dates = [datetime.now() + timedelta(days=i) for i in range(1, 101)]
times = ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00", "19:00", "20:00"]

dataset = []

# Générer 1000 exemples
for _ in range(250):
    event = random.choice(events)
    date = random.choice(dates).strftime("%Y-%m-%d")
    time = random.choice(times)

    # Générer des exemples pour chaque intention
    dataset.append((random.choice(add_event_phrases).format(event=event, date=date, time=time), "add_event"))
    dataset.append((random.choice(delete_event_phrases).format(event=event, date=date), "delete_event"))
    dataset.append((random.choice(update_event_phrases).format(event=event, time=time), "update_event"))
    dataset.append((random.choice(get_event_phrases).format(event=event, date=date), "get_event"))

# Sauvegarder le dataset en CSV
with open('event_commands.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(["command", "label"])
    for command, label in dataset:
        writer.writerow([command, label])

print(f"Generated dataset with {len(dataset)} examples.")
