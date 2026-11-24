# 02 Manuelle Bewertung der SAM-3-Masken und Qualitätsfilter (08.11.2026)

**Frage:** Wie viele SAM-3-Masken sind fürs Training brauchbar, welche Mängel haben die übrigen, und welcher Filter sortiert sie aus?

**Aufbau:** SAM 3 mit „bee“ auf 120 Bildern (je 1 pro Sekunde aus Gelb `seq015`, `seq041` und Rot `seq001`, `seq005`), zusammen 9.879 Masken. Daraus je Aufnahme 150 zufällige Masken, die ein Teammitglied manuell in `sehr_gut`, `gut` oder `schlecht/<grund>` einsortiert hat. Auf dieser Bewertung wurden feste Filterregeln geprüft (Version 1 mit festen Schwellen, Version 2 mit Schwellen relativ zum typischen Wert je Aufnahme), dazu ein gelernter Filter (DINOv2-Merkmale und logistische Regression).

**Ergebnis:**
- Ohne Filter sind 43 % (Gelb, 64 von 150) und 49 % (Rot, 74 von 150) der Masken gut oder sehr gut.
- Häufigste Mängel sind Unschärfe (Gelb 40, Rot 23 von 150, also 27 bzw. 15 %), Blick von vorn oder von der Seite (21 bzw. 27, also 14 bzw. 18 %) und Masken über mehrere Bienen (18 bzw. 11, also 12 bzw. 7 %).
- Die Sicherheit (Konfidenz) von SAM 3 trennt brauchbare und unbrauchbare Masken am besten, mit einer AUC von 0,7 bis 1,0 für fast alle Fehlerarten (`ergebnisse_version2/trennschaerfe.csv`).
- Mit der besten festen Regel sind 61 % (Gelb) bzw. 73 % (Rot) der behaltenen Masken gut, mit dem gelernten Filter 69 bzw. 87 %. Ob das auf neuen Masken hält, prüft Versuch 03.

**Im Paper:** Abschnitt 2 (Anteile und Mängel), Abschnitt 3 (Konfidenz und AUC, Wahl von SAM 3).

| Datei | Inhalt |
|---|---|
| `bewertung/gelb/`, `bewertung/rot/` | die 300 manuell bewerteten Masken. Der Ordnername ist die Bewertung |
| `notebooks/notebook_sam3_filter.ipynb`, `notebook_sam3_filter_v2.ipynb` | genau so auf Kaggle ausgeführt (Version 1 und 2) |
| `ergebnisse_version1/` | Bericht, Trichter (wie viele Masken jeder Filter entfernt), Bögen je Filter, Übersichtsbilder |
| `ergebnisse_version2/trennschaerfe.csv` | AUC je Messwert und Fehlerart (0,5 = Zufall, 1 = perfekt) |
| `ergebnisse_version2/optimierung.csv`, `einzelfilter.csv`, `trichter_varianten.csv`, `grund_vs_filter_*.csv` | Filtervarianten und welche Fehler sie treffen |
| `ergebnisse_version2/bienen_messwerte.csv` | Messwerte aller 9.879 Masken |
| `ergebnisse_version2/klassifikator/` | gelernter Filter mit Vergleich zu den Regeln, Lernkurve und Fehlerbildern |
| `ergebnisse_version2/boegen/mittel/`, `uebersicht/` | Bögen der Variante „mittel“, Übersichten aller Varianten |

**Nachrechnen:** Die Anteile oben lassen sich direkt aus den Ordnern in `bewertung/` auszählen. Die vollständige Auswertung macht `python code/lokal/filter_auswertung.py <rohdaten> <ausgabe> --optimieren` mit dem Bewertungsordner als `<ausgabe>/bewerten`. Die Rohdaten (`rohdaten.zip` mit Brettbildern und Masken) waren eine Ausgabe des Kaggle-Notebooks und liegen nicht im Repo.

Nicht im Repo sind die 9.879 einzelnen Maskenbilder, die Bögen der Varianten „locker“ und „streng“ und 22 doppelte Dateikopien aus dem Bewertungsordner, die die Auswertung ohnehin übersprungen hat.
