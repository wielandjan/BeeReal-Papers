# Paper 1: Datenpartitionierung und Labeling per Tracking zur Re-Identifikation von Honigbienen

Das Paper: [Datenpartitionierung_und_Labeling_per_Tracking_zur_Re-Identifikation_von_Honigbienen.pdf](Datenpartitionierung_und_Labeling_per_Tracking_zur_Re-Identifikation_von_Honigbienen.pdf), abgegeben am 24.11.2026.

Dieser Ordner belegt, was im Paper steht. Er enthält den Code, die Ausgaben der Läufe auf Kaggle und die manuellen Auswertungen. Nach der Abgabe wird hier nichts mehr geändert.

## So ist der Ordner aufgebaut

```
paper1/
  README.md               dieses Leseverzeichnis
  Datenpartitionierung_….pdf
  code/                   der gesamte Code, siehe code/README.md
    kaggle/               Läufe mit GPU auf Kaggle
    lokal/                Auswertungen am eigenen Rechner
  versuche/               je Versuch ein Ordner, in zeitlicher Reihenfolge
    NN_name/
      README.md           Frage, Aufbau, Ergebnis, Stelle im Paper, Dateien
      notebooks/          das Notebook genau so, wie es auf Kaggle lief (wo erhalten)
      ergebnisse/         Ausgaben des Laufs (CSV, JSON, Log, Bilder)
      bewertung/          manuell bewertete Bilder (nur Versuche 02 und 03)
```

Zum Einstieg zuerst das PDF lesen, dann die Tabelle unten. Jede Zeile führt zu dem Versuch, der eine Aussage belegt.

## Begriffe

- **Gelb / Rot:** die zwei Stöcke, Aufnahme 2123 (Gelb, 75 Sequenzen) und 2125 (Rot, 21 Sequenzen). Eine Sequenz ist ein Video von 30 s.
- **Track:** Kette von Kästen, die der Tracker im Video derselben Biene zuordnet. Im Training gilt ein Track als eine Biene.
- **Puffer:** eine ungenutzte Sequenz zwischen zwei Bereichen der Partitionierung.
- **Fold:** einer der drei Durchgänge der geblockten Kreuzvalidierung. Jeder Block ist einmal Validierung.
- **Rank-1:** Anteil der Anfragebilder, deren ähnlichstes Galeriebild dieselbe Biene zeigt. Raten ergibt 9 %.
- **Positions-Tracker / ByteTrack-Prinzip:** Der Positions-Tracker ordnet Kästen nur nach dem Abstand zu. Beim ByteTrack-Prinzip starten nur sichere Kästen (ab 0,7) neue Tracks, unsichere (ab 0,3) setzen nur fort.

## Die Versuche

| Ordner | Datum | Frage | Ergebnis |
|---|---|---|---|
| [01_detektion_grounding_dino](versuche/01_detektion_grounding_dino/README.md) | 07.11. | Wie viele Bienen findet Grounding DINO, wie lang werden die Tracks? | Median 82,5 Bienen pro Bild (Gelb), Tracks unter 1 s |
| [02_masken_bewertung_und_filter](versuche/02_masken_bewertung_und_filter/README.md) | 08.11. | Wie viele SAM-3-Masken sind brauchbar, welche Filter helfen? | ohne Filter 43 / 49 % brauchbar, Konfidenz trennt am besten |
| [03_blindtest_filter](versuche/03_blindtest_filter/README.md) | 09.11. | Halten die Filter auf neuen Masken? | gelernter Filter 41 → 67 % (Gelb), 55 → 86 % (Rot) |
| [04_tracking_abrisse_und_kleben](versuche/04_tracking_abrisse_und_kleben/README.md) | 09./10.11. | Lassen sich abgerissene Tracks verbinden? | nur 88 von 504 Verbindungen eindeutig |
| [05_tracking_bildrate](versuche/05_tracking_bildrate/README.md) | 10.11. | Helfen 30 statt 10 Bilder pro Sekunde? | nein |
| [06_tracking_sam3_und_bytetrack](versuche/06_tracking_sam3_und_bytetrack/README.md) | 11./12.11. | SAM 3 oder Grounding DINO, welcher Tracker, Videomodus? | ByteTrack-Prinzip verlängert den Median von 1,3 auf 4,4 s (Gelb), Videomodus passt nicht auf eine T4 |
| [07_tracking_sam2](versuche/07_tracking_sam2/README.md) | 14.11. | Ist SAM 2.1 Tiny als Videotracker besser? | nein, kürzere Tracks |
| [08_sam3_alle_sequenzen](versuche/08_sam3_alle_sequenzen/README.md) | 13./14.11. | Trainingslabels für alle 69 Sequenzen | Median der Track-Länge 4,7 s (Gelb), 3,5 s (Rot) |
| [09_training_filter_folds](versuche/09_training_filter_folds/README.md) | 14./15.11. | Hilft ein Qualitätsfilter im Training über alle Folds? | nein, nur in 3 von 6 Läufen besser |
| [10_annotationsplan](versuche/10_annotationsplan/README.md) | 13.–16.11. | Vorbereitung der Val-GT | sechs Ausschnitte à 15 s, 5 Bilder/s |

## Aussage im Paper und Beleg

| Abschnitt | Aussage im Paper | Beleg |
|---|---|---|
| 2 | Eine Biene ist im Median bei Gelb rund 37 px groß, bei Rot rund 61 px | [code/lokal/bienengroesse.py](code/lokal/bienengroesse.py) auf den Detektionen von [08](versuche/08_sam3_alle_sequenzen/README.md) (Gelb 37 px, Rot 60 bis 61 px) |
| 2 | Ohne Filter 43 % (Gelb) und 49 % (Rot) brauchbar, Mängel Unschärfe 27 / 15 %, Blickwinkel 14 / 18 %, mehrere Bienen 12 / 7 % | [02](versuche/02_masken_bewertung_und_filter/README.md), Ordner `bewertung/` |
| 2, Tab. 1 | Drei Blöcke und ein Testbereich je Aufnahme, getrennt durch Puffer | [code/lokal/partitionierung.py](code/lokal/partitionierung.py), dieselben Blöcke in [code/kaggle/beereal_screening.py](code/kaggle/beereal_screening.py) |
| 2 | Strenger Filter in Fold 1 um 0,8 Prozentpunkte besser (+0,3 bis +1,2), über drei Folds mit je zwei Seeds nur in drei von sechs Läufen | [09](versuche/09_training_filter_folds/README.md), [code/lokal/vergleich_folds.py](code/lokal/vergleich_folds.py) |
| 3 | Grounding DINO fand im Median rund 80 Bienen pro Bild (Gelb) | [01](versuche/01_detektion_grounding_dino/README.md) |
| 3 | SAM 3 bei Schwelle 0,3 Median 104 pro Bild (Gelb) | [08](versuche/08_sam3_alle_sequenzen/README.md), [06](versuche/06_tracking_sam3_und_bytetrack/README.md) |
| 3 | Mit dem Positions-Tracker zerfielen die SAM-3-Tracks stärker (556 statt 275 Tracks) | [06](versuche/06_tracking_sam3_und_bytetrack/README.md), `p2_ergebnis_einzelbild.csv` |
| 3 | Konfidenz trennt brauchbare und unbrauchbare Masken am besten (AUC 0,7 bis 1,0) | [02](versuche/02_masken_bewertung_und_filter/README.md), `trennschaerfe.csv` |
| 3 | 30 Bilder pro Sekunde ergaben keine längeren Tracks | [05](versuche/05_tracking_bildrate/README.md) |
| 3 | Bei 82 bis 92 % der Track-Enden beginnt binnen 6 s nah ein neuer Track | [06](versuche/06_tracking_sam3_und_bytetrack/README.md), `p2_ergebnis_einzelbild.csv`, Zeilen der 30-s-Clips |
| 3 | Verbinden über 6 s nur in 88 von 504 Fällen eindeutig | [04](versuche/04_tracking_abrisse_und_kleben/README.md) |
| 3 | ByteTrack-Prinzip verlängert den Median von 1,3 auf 4,4 s (Gelb) und von 1,4 auf 3,7 s (Rot), Startschwelle 0,5 nur 2,1 s | [06](versuche/06_tracking_sam3_und_bytetrack/README.md), `bytetrack_ergebnis.csv` |
| 3 | Über alle 69 Sequenzen 4,7 bzw. 3,5 s | [08](versuche/08_sam3_alle_sequenzen/README.md), `tracks_statistik.csv` |
| 3, Tab. 2 | SAM-3-Video stabiler (Rot, 5 s, Median 4,9 statt 1,7 s), auf T4 nur 1,2 bis 5 s | [06](versuche/06_tracking_sam3_und_bytetrack/README.md), `info_v3_video.json` |
| Tab. 2 | SAM 2.1 Tiny kürzer | [07](versuche/07_tracking_sam2/README.md) |
| 3, Tab. 2 | Weicher und strenger Filter ohne Gewinn in allen Folds | [09](versuche/09_training_filter_folds/README.md) |
| 3 | Gelernter Filter im Blindtest 41 → 67 % (Gelb), 55 → 86 % (Rot) | [03](versuche/03_blindtest_filter/README.md) |
| 4 | Val-GT mit sechs Ausschnitten von 15 s, 5 Bilder pro Sekunde, Vorschläge aus dem Tracking | [10](versuche/10_annotationsplan/README.md) |

## Was nicht hier liegt

- Die Videos. Sie stammen aus dem Datensatz *thehiveisreal* (Hugging Face, `mrkschtr/thehiveisreal`, CC BY-NC 4.0). Die Bienenbilder in diesem Ordner sind Ausschnitte daraus und stehen unter derselben Lizenz.
- Trainierte Modelle und Merkmale, weil sie zu groß sind.
- Die fertige Ground Truth, ihre Auswertung und der Vergleich der Re-ID-Modelle. Sie gehören zu Paper 2.
- Die Notebooks der Versuche 01, 06 und 07 sind nicht als Datei erhalten. Dort liegt nur das Skript in `code/kaggle/`.
