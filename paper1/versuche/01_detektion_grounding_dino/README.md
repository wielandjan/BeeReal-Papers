# 01 Grounding DINO und Positions-Tracker (07.11.2026)

**Frage:** Wie viele Bienen findet Grounding DINO pro Bild, und wie lang werden die Tracks mit einem einfachen Positions-Tracker?

**Aufbau:** Grounding DINO mit der Textanweisung „bee“ auf 46 Sequenzen (Gelb und Rot), jedes 6. Bild (10 Bilder/s), im festen Ausschnitt des Flugbretts. Der Positions-Tracker ordnet Kästen aufeinanderfolgender Bilder nach dem Abstand ihrer Mitten zu (Abstand unter 0,5 Bienengrößen, Lücke bis 5 Bilder). Code: [code/kaggle/beereal_baseline.py](../../code/kaggle/beereal_baseline.py). Der Lauf trainierte danach auch ein Re-ID-Modell, das gehört zu Paper 2 und liegt nicht hier.

**Ergebnis:** Im Median 82,5 Bienen pro Bild bei Gelb und 44,7 bei Rot. Die Tracks sind kurz (Median 7 bzw. 8 Bilder, also unter 1 s) und reißen ständig ab.

**Im Paper:** Abschnitt 3, „In Vorversuchen fand es im Median rund 80 Bienen pro Bild (Gelb)“.

| Datei | Inhalt |
|---|---|
| `ergebnisse/sequence_stats.csv` | je Sequenz Bienen pro Bild, Tracks, Median der Track-Länge in Bildern. Die Spalte `split` ist die alte feste Aufteilung, die später verworfen wurde |
| `ergebnisse/roi_gelb.png`, `roi_rot.png`, `roi_rot_seq001_pruefung.png` | der feste Ausschnitt des Flugbretts je Aufnahme |
| `ergebnisse/tracks_gelb.png`, `tracks_rot.png` | Beispiel der Tracks im Bild |

Das Notebook dieses Laufs ist nicht erhalten. `beereal_baseline.py` ist der spätere Stand des Skripts, Detektion und Tracker sind darin unverändert.
