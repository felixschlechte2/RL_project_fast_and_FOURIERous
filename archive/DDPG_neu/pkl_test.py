import pickle

file = r''

# Datei öffnen und laden
with open(file, 'rb') as file:
    data = pickle.load(file)

# Inhalt anzeigen
# print(data)

# Falls du den Typ des Objekts wissen möchtest
print(type(data))
