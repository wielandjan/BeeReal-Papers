# Code zu Paper 1

Jede Datei beginnt mit einer Zeile, zu welchem Versuch sie gehört, danach folgt die Beschreibung mit Ein- und Ausgabe und Aufruf.

- `kaggle/` enthält die Skripte, die auf Kaggle mit GPU liefen (2× Nvidia T4). Die Zellen sind durch `# %%` getrennt, `lokal/make_nb.py` baut daraus ein Notebook. Wo das ausgeführte Notebook erhalten ist, ist der Code genau aus diesem Notebook übernommen. Das Notebook selbst liegt im Versuchsordner.
- `lokal/` enthält die Auswertungen am eigenen Rechner (CPU). Sie bekommen alle Pfade als Argumente.

Den Zugang zu Hugging Face lesen die Kaggle-Skripte aus einem privaten Kaggle-Datensatz (`hf_token`). Den Zugang zu CVAT liest `cvat_hochladen.py` aus einer Datei im Benutzerordner. Beides liegt nicht im Repo.

## Pipeline, wie sie im Paper steht

| Schritt | Datei | Versuch |
|---|---|---|
| Partitionierung festlegen und prüfen | [lokal/partitionierung.py](lokal/partitionierung.py) | Abschnitt 2, Tab. 1 |
| Bienen finden mit SAM 3 („bee“, Schwelle 0,3, 10 Bilder/s) | [kaggle/beereal_stufe_b_sam3.py](kaggle/beereal_stufe_b_sam3.py) | 08 |
| Tracks nach dem ByteTrack-Prinzip (Start ab 0,7, Fortsetzen ab 0,3, Lücke bis 0,5 s) | [lokal/stufe_b_tracks.py](lokal/stufe_b_tracks.py) | 08 |
| Ausschnitte je Track schneiden | [kaggle/beereal_ausschnitte.py](kaggle/beereal_ausschnitte.py) | 09 |
| Training über drei Folds, Rank-1 nach Chan et al. | [kaggle/beereal_screening.py](kaggle/beereal_screening.py) | 09 |
| Gepaarter Vergleich und Siegregel | [lokal/vergleich_folds.py](lokal/vergleich_folds.py) | 09 |
| Vorschläge für die Val-GT | [kaggle/beereal_val_gt_sam3.py](kaggle/beereal_val_gt_sam3.py), [lokal/cvat_paket.py](lokal/cvat_paket.py), [lokal/cvat_sammeln.py](lokal/cvat_sammeln.py), [lokal/cvat_hochladen.py](lokal/cvat_hochladen.py) | 10 |

## Vergleiche und Vorversuche

| Datei | Was | Versuch |
|---|---|---|
| [kaggle/beereal_baseline.py](kaggle/beereal_baseline.py) | Grounding DINO und Positions-Tracker. Enthält `track()`, das mehrere lokale Skripte laden | 01 |
| [lokal/bienengroesse.py](lokal/bienengroesse.py) | typische Bienengröße je Aufnahme | Abschnitt 2 |
| [kaggle/beereal_sam3_filter.py](kaggle/beereal_sam3_filter.py) | SAM-3-Masken ausschneiden und messen | 02 |
| [lokal/filter_auswertung.py](lokal/filter_auswertung.py) | Filtervarianten, Bewertungsordner, Auswertung der manuellen Bewertung | 02, 03 |
| [lokal/qualitaet_klassifikator.py](lokal/qualitaet_klassifikator.py) | gelernter Filter | 02 |
| [kaggle/beereal_blindtest.py](kaggle/beereal_blindtest.py), [lokal/blindtest.py](lokal/blindtest.py) | Blindtest der Filter auf neuen Masken | 03 |
| [lokal/tracks_kleben.py](lokal/tracks_kleben.py), [lokal/klebestellen_bogen.py](lokal/klebestellen_bogen.py) | abgerissene Tracks verbinden, Prüfbögen | 04 |
| [kaggle/beereal_fps_test.py](kaggle/beereal_fps_test.py), [lokal/fps_auswertung.py](lokal/fps_auswertung.py) | Bildrate 30, 15, 10 Bilder/s | 05 |
| [kaggle/beereal_p2_sam3video.py](kaggle/beereal_p2_sam3video.py), [lokal/p2_auswertung.py](lokal/p2_auswertung.py) | SAM 3 Einzelbild und Videomodus | 06 |
| [lokal/bytetrack_test.py](lokal/bytetrack_test.py) | ByteTrack-Prinzip gegen Positions-Tracker auf denselben Kästen | 06 |
| [kaggle/beereal_sam2_tracker.py](kaggle/beereal_sam2_tracker.py) | SAM 2.1 Tiny als Videotracker | 07 |
| [lokal/make_nb.py](lokal/make_nb.py), [lokal/kg.py](lokal/kg.py) | Notebook bauen, Kaggle-Befehlszeile | Hilfsmittel |

## Benötigte Pakete

Lokal: `numpy`, `pandas`, `scipy`, `opencv-python`, `pillow`, `matplotlib`, für den gelernten Filter zusätzlich `torch`, `transformers`, `scikit-learn`, für CVAT `cvat-sdk`. Auf Kaggle installieren die Skripte, was fehlt.
