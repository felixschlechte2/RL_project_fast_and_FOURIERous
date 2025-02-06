import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Pfad zur TensorBoard-Protokolldatei (ersetze den Pfad entsprechend)
log_file_path = './runs/2025-01-29_00-52-13_SAC_hockey_vs_rand_all_rew_Gaussian_/events.out.tfevents.1738108333.0'

# EventAccumulator initialisieren
event_acc = EventAccumulator(log_file_path)
event_acc.Reload()  # Daten aus der Datei laden: kann einige minuten dauern!

# Tags (verfügbare Kategorien) abrufen
print("Verfügbare Tags:", event_acc.Tags())

# Beispiel: Skalare visualisieren (z. B. 'loss')
if 'scalars' in event_acc.Tags():
    scalars = event_acc.Tags()['scalars']  # Alle verfügbaren Skalare
    for scalar in scalars:
        # Daten für jeden Skalar abrufen
        scalar_data = event_acc.Scalars(scalar)

        # Schritte und Werte extrahieren
        steps = [event.step for event in scalar_data]
        values = [event.value for event in scalar_data]

        # Plot erstellen
        plt.plot(steps, values, label=scalar)

# Plot anpassen und anzeigen
plt.xlabel('Steps')
plt.ylabel('Value')
plt.title('Scalars from TensorBoard Logs')
plt.legend()
plt.show()
