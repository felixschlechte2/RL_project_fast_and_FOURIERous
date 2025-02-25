#!/bin/bash
# Durchlaufe alle Dateien, die mit dem gegebenen Muster beginnen
for file in sac_buffer_hockey_vs_strong_and_self_all_rew_*; do
    # Extrahiere den nummerischen Teil nach dem letzten "_ep_"
    num=${file##*_rew_}
    
    # Stelle sicher, dass der extrahierte Teil ausschließlich aus Ziffern besteht
    if [[ $num =~ ^[0-9]+$ ]]; then
        # Prüfe, ob num kein Vielfaches von 1000 ist
        if (( num % 5000 == 0 )) && (( num != 30000 )); then
            echo "Lösche Datei: $file"
            rm "$file"
        else
            echo "Behalte Datei: $file (Vielfaches von 1000)"
        fi
    else
        echo "Datei $file übersprungen, da der nummerische Teil nicht erkannt wurde."
    fi
done

