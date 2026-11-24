# 03 Blindtest der Qualitätsfilter (09.11.2026)

**Frage:** Hält der Vorsprung der Filter auf neuen Masken, die bei der Einstellung nicht vorkamen?

**Aufbau:** 200 neue Zufallsmasken (je 100 pro Aufnahme) aus Gelb `seq030`, `seq052` und Rot `seq003`, `seq007`, je 15 Bilder im Abstand von 2 s. Beide Filter wurden nur aus der Bewertung von Versuch 02 eingestellt bzw. gelernt. Die Masken wurden manuell bewertet, ohne die Entscheidung der Filter zu kennen. Diese steht in `filter_entscheidung_NICHT_VOR_DEM_BEWERTEN_OEFFNEN.csv`.

**Ergebnis:** Anteil guter Masken unter den behaltenen, in Klammern der 95-%-Bereich.

| | Gelb | Rot |
|---|---|---|
| ohne Filter | 41 % | 55 % |
| feste Regel | 62 % (48–75) | 82 % (70–93) |
| gelernter Filter | 67 % (52–80) | 86 % (74–96) |

Beide Filter wirken, der gelernte ist nicht klar besser. Im Training wurde der gelernte Filter nicht getestet (siehe Versuch 09).

**Im Paper:** Abschnitt 3, „Ein gelernter Filter hob in einem Blindtest auf 200 neuen Masken den Anteil brauchbarer Masken von 41 auf 67 % (Gelb) und von 55 auf 86 % (Rot)“.

| Datei | Inhalt |
|---|---|
| `bewertung/` | die 200 bewerteten Masken in Ordnern |
| `notebooks/notebook_blindtest.ipynb` | genau so auf Kaggle ausgeführt |
| `ergebnisse/blindtest_ergebnis.csv` | die Tabelle oben |
| `ergebnisse/blindtest_uneinig.csv`, `blindtest_grund.csv` | wo sich die Filter uneinig sind, welche Fehler sie treffen |
| `ergebnisse/bienen_messwerte.csv`, `einzelfilter.csv`, `trichter_varianten.csv` | Messwerte und Filter auf allen 4.772 neuen Masken |

**Nachrechnen:** `python code/lokal/blindtest.py auswerten <ordner>`, wobei `<ordner>` die Dateien aus `ergebnisse/` und den Bewertungsordner als `bewerten/` enthält.
