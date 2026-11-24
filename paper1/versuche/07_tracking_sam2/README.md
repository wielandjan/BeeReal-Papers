# 07 SAM 2.1 Tiny als Videotracker (14.11.2026)

**Frage:** Verfolgt ein kleiner Videotracker mit Gedächtnis die Bienen besser als das ByteTrack-Prinzip, und passt er auf eine T4?

**Aufbau:** Die sechs Val-Ausschnitte aus Versuch 10 (je 15 s, 5 Bilder/s). In Fenstern von 3 s dienen die sicheren SAM-3-Kästen (ab 0,7) als Startkästen, SAM 2.1 Tiny verfolgt sie durch das Fenster, benachbarte Fenster werden über eine IoU ab 0,5 verbunden.

**Ergebnis:** Läuft (4 bis 5 GB, 4 bis 7 min pro Ausschnitt), liefert aber weniger Kästen pro Bild (Gelb rund 61 statt 84, Rot 33 statt 40) und weniger lange Tracks (ab 10 s bei Gelb 22 bis 29 statt 55 bis 66, bei Rot 15 bis 20 statt 18 bis 29). Das Verbinden an den Fenstergrenzen reißt oft ab. In dieser Form ist SAM 2.1 nicht besser als das ByteTrack-Prinzip.

**Im Paper:** Tab. 2, Zeile Videotracker („SAM 2.1 kürzer“).

| Datei | Inhalt |
|---|---|
| `ergebnisse/sam2_<sequenz>.json` | Tracks von SAM 2.1 je Ausschnitt |
| `ergebnisse/beereal-sam2-tracker.log` | Kaggle-Log mit Laufzeit und Speicher |

**Code:** [code/kaggle/beereal_sam2_tracker.py](../../code/kaggle/beereal_sam2_tracker.py). Das Notebook ist nicht als eigene Datei erhalten.
