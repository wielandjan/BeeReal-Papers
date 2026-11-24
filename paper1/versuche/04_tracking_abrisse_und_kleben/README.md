# 04 Abgerissene Tracks nachträglich verbinden (09./10.11.2026)

**Frage:** Lassen sich abgerissene Tracks sicher verbinden, wenn die Biene nur kurz nicht erkannt wurde?

**Aufbau:** Tracks des Positions-Trackers auf Gelb `seq065` und Rot `seq012` (Grounding DINO, 30 s). Ein Track-Ende wird mit einem späteren Track-Anfang verbunden, wenn dieser höchstens 6 s später und weniger als eine Bienenlänge entfernt beginnt (lineare Zuordnung, drei Runden). Bei „streng“ muss der zweitbeste Kandidat mindestens doppelt so weit weg sein, die Verbindung also eindeutig.

**Ergebnis:**

| | Tracks vorher | streng | ohne Eindeutigkeit |
|---|---|---|---|
| Gelb seq065 | 704 | 616 (88 Verbindungen) | 200 (504 Verbindungen) |
| Rot seq012 | 352 | 309 (43 Verbindungen) | 137 (215 Verbindungen) |

Rechnerisch ließen sich bei Gelb 504 Enden verbinden, eindeutig sind nur 88, weil fast immer eine zweite Biene in Reichweite liegt. Streng verbunden werden die Tracks kaum länger (Tracks ab 10 s bei Gelb 83 auf 86). Auf den Prüfbögen ist bei „ohne Eindeutigkeit“ oft die Nachbarbiene zu sehen.

**Im Paper:** Abschnitt 3, „Ein Verbinden über bis zu 6 s war wegen benachbarter Bienen nur in 88 von 504 Fällen eindeutig (Gelb, eine Sequenz)“.

| Datei | Inhalt |
|---|---|
| `ergebnisse/statistik_streng.csv`, `statistik_ohne_eindeutigkeit.csv` | Tracks vorher und nachher, Anzahl Verbindungen |
| `ergebnisse/bogen_*_gelb.png`, `bogen_*_rot.png` | Prüfbögen, links das Bild am Ende des alten Tracks, rechts am Anfang des neuen |

**Code:** [code/lokal/tracks_kleben.py](../../code/lokal/tracks_kleben.py) und [code/lokal/klebestellen_bogen.py](../../code/lokal/klebestellen_bogen.py). Die Detektionen der beiden Sequenzen lagen nur lokal und sind nicht im Repo.
