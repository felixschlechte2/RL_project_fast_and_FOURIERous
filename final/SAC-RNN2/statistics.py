import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle as pkl

class statistics():
    def __init__(self, attributes, saving_path):
        self.stats = {}
        for attr in attributes:
            self.stats[attr] = []
        self.saving_path = saving_path
        self.save_statistics()

    def save_data(self, attributes, values):
        if len(attributes) != len(values):
            print("error: both list must be of equal length!")
            return
        else:
            for i in range(len(attributes)):
                self.stats[attributes[i]].append(values[i])
    def save_statistics(self):
        with open(self.saving_path, "wb") as f:
            pkl.dump(self.stats, f)
    def load_statistics(self, path):
        with open(path, "rb") as f:
            self.stats = pkl.load(f)
    def return_values(self, attribute):
        return self.stats[attribute]

test = statistics(["A", "B"], f"./test.pkl")

for i in range(10):
    test.save_data(["A", "B"], [i, 2*i])

test.save_statistics()

test.load_statistics(f"./test.pkl")

plt.plot(test.return_values("B"), marker='o', linestyle='-', color='b', label="...")

# Achsenbeschriftungen und Titel
plt.xlabel("..")
plt.ylabel("...")
plt.title("..")

# Legende anzeigen
plt.legend()

# Grid hinzufügen (optional)
plt.grid(True)

# Plot anzeigen
plt.show()

###################################### stats with pickle:
# data_path = fr"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/QR-SAC4/runs/stats_hockey_vs_random_all_rew-qr_sac_run4.pkl"
# data_path = fr"./Reinforcement Learning/ExcercisesGitHub/exercises mit venv/project_code/models/DDPG_alt/results/DDPG_hockey-eps0.1-t50-l0.0001-sNone-stat.pkl"
# with open(data_path, "rb") as f:
#     data = pkl.load(f)

# plt.plot(data['rewards'], marker='o', linestyle='-', color='b', label="QR-SAC vs random")

# # Achsenbeschriftungen und Titel
# plt.xlabel("testing every ... epoch")
# plt.ylabel("mean reward over 10 testing epochs")
# plt.title("reward/test")

# # Legende anzeigen
# plt.legend()

# # Grid hinzufügen (optional)
# plt.grid(True)

# # Plot anzeigen
# plt.show()

####################################### stats wih tensorboard:
# Pfad zur TensorBoard-Protokolldatei (ersetze den Pfad entsprechend)
# log_file_path = './runs/2025-01-29_00-52-13_SAC_hockey_vs_rand_all_rew_Gaussian_/events.out.tfevents.1738108333.0'

# # EventAccumulator initialisieren
# event_acc = EventAccumulator(log_file_path)
# event_acc.Reload()  # Daten aus der Datei laden: kann einige minuten dauern!

# # Tags (verfügbare Kategorien) abrufen
# print("Verfügbare Tags:", event_acc.Tags())

# # Beispiel: Skalare visualisieren (z. B. 'loss')
# if 'scalars' in event_acc.Tags():
#     scalars = event_acc.Tags()['scalars']  # Alle verfügbaren Skalare
#     for scalar in scalars:
#         # Daten für jeden Skalar abrufen
#         scalar_data = event_acc.Scalars(scalar)

#         # Schritte und Werte extrahieren
#         steps = [event.step for event in scalar_data]
#         values = [event.value for event in scalar_data]

#         # Plot erstellen
#         plt.plot(steps, values, label=scalar)

# # Plot anpassen und anzeigen
# plt.xlabel('Steps')
# plt.ylabel('Value')
# plt.title('Scalars from TensorBoard Logs')
# plt.legend()
# plt.show()
