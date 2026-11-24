# 06 SAM 3 gegen Grounding DINO, ByteTrack-Prinzip und Videomodus (11./12.11.2026)

Drei Fragen auf denselben Clips (Gelb `seq065`, Rot `seq012`, 10 Bilder/s).

## Detektor: Grounding DINO oder SAM 3 im Einzelbildmodus?

Mit dem Positions-Tracker, erste 10 s:

| Clip | Verfahren | Bienen pro Bild | Tracks | Median Track-Länge |
|---|---|---|---|---|
| Gelb | Grounding DINO | 78 | 275 | 1,3 s |
| Gelb | SAM 3, Schwelle 0,3 | 104 | 556 | 1,0 s |
| Gelb | SAM 3, Schwelle 0,5 | 80 | 302 | 1,3 s |
| Rot | Grounding DINO | 34 | 116 | 1,45 s |
| Rot | SAM 3, Schwelle 0,3 | 42 | 285 | 0,7 s |
| Rot | SAM 3, Schwelle 0,5 | 35 | 135 | 1,4 s |

SAM 3 findet mehr Bienen, mit dem Positions-Tracker zerfallen die Tracks aber stärker. Über die vollen 30 s mit Grounding DINO beginnt bei 92 % (Gelb) bzw. 82 % (Rot) der Track-Enden binnen 6 s weniger als eine Bienenlänge entfernt ein neuer Track. Die Tracks reißen also ab, weil der Detektor die Biene verliert.

## Tracker: ByteTrack-Prinzip auf denselben SAM-3-Kästen

Kästen ab der Startschwelle beginnen Tracks und setzen sie fort, Kästen ab 0,3 setzen nur fort.

| Clip | Tracker | Tracks | Median Track-Länge |
|---|---|---|---|
| Gelb | Positions-Tracker (ab 0,5) | 302 | 1,3 s |
| Gelb | ByteTrack-Prinzip (Start 0,7, Fortsetzen 0,3) | 169 | 4,4 s |
| Gelb | ByteTrack-Prinzip (Start 0,5, Fortsetzen 0,3) | 273 | 2,1 s |
| Rot | Positions-Tracker (ab 0,5) | 135 | 1,4 s |
| Rot | ByteTrack-Prinzip (Start 0,7, Fortsetzen 0,3) | 73 | 3,7 s |

## Videomodus von SAM 3

Auf einer Nvidia T4 (14,5 GB) läuft der Videomodus bei Rot nur 5 s (50 Bilder, 9,2 GB) und bei Gelb nur 1,2 s (12 Bilder), danach ist der Speicher voll. Wo er läuft (Rot, erste 5 s), hält er die Bienen stabiler, mit 41 Tracks und Median 4,9 s gegen 74 Tracks und Median 1,65 s bei Grounding DINO und Positions-Tracker.

**Im Paper:** Abschnitt 3, erster und zweiter Absatz, und Tab. 2 (Zeilen Detektion, Tracking, Videotracker).

| Datei | Inhalt |
|---|---|
| `ergebnisse/p2_ergebnis_einzelbild.csv` | Kennzahlen der Detektoren mit Positions-Tracker, auch für die 30-s-Clips |
| `ergebnisse/bytetrack_ergebnis.csv` | ByteTrack-Prinzip mit verschiedenen Schwellen |
| `ergebnisse/bild_*_10s.json` | SAM-3-Kästen im Einzelbildmodus |
| `ergebnisse/video_*.json`, `ok_*.json`, `info_v1.json`, `info_v3_video.json` | Videomodus mit Ausgaben, Zeiten, Speicher und Fehlermeldungen |
| `ergebnisse/bytetrack_stellen_*.json`, `pruefbogen_*.png` | Stellen, an denen ein unsicherer Kasten einen Track fortsetzt, mit Prüfbögen |

**Code:** [code/kaggle/beereal_p2_sam3video.py](../../code/kaggle/beereal_p2_sam3video.py) (Kaggle), [code/lokal/p2_auswertung.py](../../code/lokal/p2_auswertung.py), [code/lokal/bytetrack_test.py](../../code/lokal/bytetrack_test.py).

**Nachrechnen ByteTrack:** `python code/lokal/bytetrack_test.py code/kaggle/beereal_baseline.py versuche/06_tracking_sam3_und_bytetrack/ergebnisse <ausgabe.csv>`
