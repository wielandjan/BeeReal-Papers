# 10 Vorbereitung der Validierungs-Ground-Truth (13.–16.11.2026)

**Was:** Je Block ein Ausschnitt von 15 s, also sechs Ausschnitte (Gelb `seq021`, `seq036`, `seq073`, Rot `seq001`, `seq005`, `seq012`). Annotiert werden 5 Bilder pro Sekunde, also 75 Bilder je Ausschnitt. Die Vorschläge kommen aus SAM 3 und dem ByteTrack-Prinzip und werden manuell korrigiert. Die Test-Ground-Truth entsteht später ohne Vorschläge.

**Im Paper:** Abschnitt 4.

| Datei | Inhalt |
|---|---|
| `vorschlaege_sam3/bild_<sequenz>_0-15s.json` | SAM-3-Kästen je Bild der sechs Ausschnitte |
| `pakete_val/<sequenz>_0-15s_info.json` | je Ausschnitt Start, Dauer, Bilder pro Sekunde und die genutzten Videobilder |
| `pakete_val/Val_GT_info.json` | die sechs Ausschnitte als ein Paket mit je 75 Bildern |
| `pakete_val/Val_GT_vorschlaege_mot.zip` | die Vorschläge mit Track-IDs im Format MOT 1.1 |

**Code:** [code/kaggle/beereal_val_gt_sam3.py](../../code/kaggle/beereal_val_gt_sam3.py), [code/lokal/cvat_paket.py](../../code/lokal/cvat_paket.py), [code/lokal/cvat_sammeln.py](../../code/lokal/cvat_sammeln.py), [code/lokal/cvat_hochladen.py](../../code/lokal/cvat_hochladen.py). Die Bilder der Ausschnitte lassen sich mit `cvat_paket.py` aus den Videos neu erzeugen und liegen nicht im Repo. Die fertige Ground Truth und ihre Auswertung gehören zu Paper 2.
