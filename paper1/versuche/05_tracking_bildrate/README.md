# 05 Mehr Bilder pro Sekunde fürs Tracking? (10.11.2026)

**Frage:** Werden die Tracks länger, wenn mehr Bilder pro Sekunde ausgewertet werden, weil sich die Bienen zwischen zwei Bildern weniger bewegen?

**Aufbau:** Grounding DINO auf den ersten 10 s von Gelb `seq065` und Rot `seq012`, jedes 2. Bild (rund 30 Bilder/s). Daraus 30, 15 und 10 Bilder/s. Der Positions-Tracker bekommt immer dieselbe Lückenzeit von 0,5 s.

**Ergebnis:** Mehr Bilder bringen keine längeren Tracks. Die Zahl der Tracks ab 5 s und ab 9 s bleibt gleich (Gelb 68 / 40 bei 30 Bildern/s gegen 70 / 39 bei 10 Bildern/s, Rot 30 / 14 gegen 29 / 13). Die Lücken sind bei allen Bildraten gleich lang (Median 0,7 bis 1,2 s), der Detektor verliert die Biene also über längere Zeit am Stück.

**Im Paper:** Abschnitt 3, „Ausgewertet wird jedes sechste Bild, also 10 Bilder pro Sekunde, denn 30 Bilder pro Sekunde ergaben in zwei Clips keine längeren Tracks.“

| Datei | Inhalt |
|---|---|
| `notebooks/notebook_bildrate.ipynb` | genau so auf Kaggle ausgeführt |
| `ergebnisse/bildrate_ergebnis.csv` | Kennzahlen je Clip und Bildrate |
| `ergebnisse/dets_*_fps30.json.gz` | die Detektionen bei 30 Bildern/s |

**Nachrechnen:** `python code/lokal/fps_auswertung.py code/kaggle/beereal_baseline.py <ausgabe> versuche/05_tracking_bildrate/ergebnisse/dets_*_fps30.json.gz`
