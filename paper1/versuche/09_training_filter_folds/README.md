# 09 Training über die drei Folds und Qualitätsfilter (14./15.11.2026)

**Frage:** Verbessert ein Qualitätsfilter für die Trainingsausschnitte das Re-ID-Training, und hält ein Ergebnis aus einem Fold über alle drei Folds?

**Aufbau:** Partitionierung wie in Tab. 1 des Papers ([code/lokal/partitionierung.py](../../code/lokal/partitionierung.py)). Die Tracks aus Versuch 08 sind die Labels, jeder Track gilt als eine Biene. Trainiert wird DINOv2-small mit einem linearen Kopf auf 128 Merkmale und Triplet-Verlust. Zuerst 1000 Schritte nur der Kopf, dann 2000 Schritte die letzten vier Blöcke, Gelb und Rot gemeinsam im Verhältnis 50:50. Gemessen wird Rank-1 wie bei Chan et al. (2022). Die Galerie hat ein Bild derselben Biene mindestens 10 s später und zehn gleichzeitig sichtbare andere Bienen, Raten ergibt 9 %. Als dieselbe Biene gilt derselbe automatische Track.

- `M_ohne`: Gelb und Rot, nur Grundfilter
- `M_weich`: Sicherheit ab 0,5 und Überlappung bis 0,2
- `M_streng`: Sicherheit ab 0,5, Überlappung bis 0,05, nicht am Rand, Größe 0,4 bis 1,8 × Median
- `G_ohne`: nur Gelb trainiert

Fold 1 lief mit allen vier Varianten und einem Seed, danach Folds 2 und 3 und ein zweiter Seed.

**Ergebnis:** Gepaarter Unterschied zu `M_ohne` auf Gelb in Prozentpunkten, mit 95-%-Bereich aus einem Bootstrap über ganze Sequenzen.

| | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| M_streng, Seed 0 | +0,8 [+0,3 bis +1,2] | +0,8 [+0,4 bis +1,3] | −0,1 [−0,4 bis +0,2] |
| M_streng, Seed 1 | −0,1 [−0,6 bis +0,3] | +0,6 [+0,4 bis +0,9] | −0,2 [−0,5 bis +0,2] |

Der strenge Filter ist auf Gelb nur in 3 von 6 Läufen besser, auf Rot in 1 von 6. Nach der Siegregel (besser in allen drei Folds) gewinnt kein Filter, es bleibt beim Grundfilter. Nur Gelb zu trainieren kostet Rot in jedem Fold 4 bis 5 Punkte.

**Im Paper:** Abschnitt 2 (Siegregel, „In Fold 1 allein war ein strenger Qualitätsfilter auf Gelb um 0,8 Prozentpunkte besser … nur in drei von sechs Läufen besser“), Abschnitt 3 und Tab. 2 (Zeile Filter).

| Datei | Inhalt |
|---|---|
| `notebooks/notebook_screening_fold1.ipynb`, `notebook_screening_folds.ipynb` | genau so auf Kaggle ausgeführt |
| `ergebnisse/fold1/`, `ergebnisse/folds2_3_seed2/` | `results.csv` mit allen Rank-1-Werten, `ergebnis_<variante>.json` mit Einstellungen, Trainingsverlauf und Rank-1 je Sequenz, Kaggle-Log |

**Nachrechnen:** `python code/lokal/vergleich_folds.py versuche/09_training_filter_folds/ergebnisse/fold1 versuche/09_training_filter_folds/ergebnisse/folds2_3_seed2` gibt die Tabelle oben für Gelb und Rot aus. Die trainierten Modelle und Merkmale sind zu groß fürs Repo.

Längeres Training und mehr feinjustierte Blöcke (Läufe vom 15. und 16.11.) gehören zu Paper 2.
