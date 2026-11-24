# 08 SAM 3 und ByteTrack-Prinzip auf allen Trainings- und Validierungssequenzen (13./14.11.2026)

**Was:** Die Trainingslabels für alle 69 Sequenzen der drei Blöcke (Gelb 57, Rot 12). Test und Puffer wurden absichtlich nicht verarbeitet.

1. SAM 3 im Einzelbildmodus, Text „bee“, Schwelle 0,3, 10 Bilder/s (300 Bilder pro Sequenz, Rot `seq000` erst ab 15 s). Laufzeit 7,27 h auf 2× T4, keine Fehler. Im Mittel 104,1 Kästen pro Bild bei Gelb und 53,2 bei Rot.
2. Tracks nach dem ByteTrack-Prinzip, lokal gerechnet. Kästen ab 0,7 starten Tracks, ab 0,3 setzen sie nur fort, Lücke bis 0,5 s, Abstand der Mitten unter 0,5 Bienengrößen.

**Ergebnis:** Der Median der Track-Länge liegt im Mittel über die Sequenzen bei 4,7 s (Gelb) und 3,5 s (Rot). Pro Sequenz gibt es rund 100 (Gelb) bzw. 47 (Rot) Tracks ab 10 s.

**Im Paper:** Abschnitt 3, „Bei Schwelle 0,3 fand es mehr Bienen (Median 104 pro Bild bei Gelb)“ und „Über alle 69 Trainings- und Validierungssequenzen liegt er bei 4,7 bzw. 3,5 s“. Grundlage für Versuch 09.

| Datei | Inhalt |
|---|---|
| `notebooks/notebook_stufe_b.ipynb` | genau so auf Kaggle ausgeführt |
| `ergebnisse/beereal-stufe-b-sam3.log`, `stufe_b_info.json` | Kaggle-Log, fertige Sequenzen |
| `ergebnisse/detektionen/sam3_<sequenz>.json.gz` | je Bild eine Liste `[sicherheit, x1, y1, x2, y2, fläche, füllgrad, überlappung]`, Koordinaten im Ausschnitt des Flugbretts |
| `ergebnisse/tracks/tracks_<sequenz>.json.gz` | die Tracks mit allen Kästen |
| `ergebnisse/tracks/tracks_statistik.csv` | je Sequenz Anzahl Tracks, Median der Länge, Tracks ab 10 s |

**Nachrechnen:**
- Tracks: `python code/lokal/stufe_b_tracks.py versuche/08_sam3_alle_sequenzen/ergebnisse/detektionen <ausgabe>`
- Bienengröße aus Abschnitt 2: `python code/lokal/bienengroesse.py versuche/08_sam3_alle_sequenzen/ergebnisse/detektionen`. Das Skript nimmt den Median der längeren Kastenseite, Gelb `seq015` 37 px, Rot `seq001` 60 px und `seq005` 61 px.
