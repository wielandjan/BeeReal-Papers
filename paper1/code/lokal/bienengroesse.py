# BeeReal Paper 1, Abschnitt 2 (Bienengröße). Nachträglich geschrieben, um die Angabe aus den Daten nachzurechnen.
"""Typische Bienengröße je Aufnahme: Median der längeren Kastenseite der SAM-3-Kästen einer Sequenz, in Pixeln.

Eingabe sind die Detektionen aus Versuch 08 (sam3_<sequenz>.json.gz, je Bild eine Liste
[sicherheit, x1, y1, x2, y2, fläche, füllgrad, überlappung]).

Aufruf:  python bienengroesse.py <ordner mit sam3_*.json.gz> [sequenz ...] [--schwelle=0.3]
Ohne Sequenzen werden rec2123_seq015 (Gelb) und rec2125_seq001 (Rot) ausgewertet.
"""
import glob, gzip, json, os, sys
import numpy as np

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
ORDNER, SEQUENZEN = POS[0], POS[1:] or ["rec2123_seq015", "rec2125_seq001"]
SCHWELLE = float(ARGS.get("schwelle", 0.3))

for sid in SEQUENZEN:
    d = json.load(gzip.open(os.path.join(ORDNER, f"sam3_{sid}.json.gz"), "rt"))
    k = np.array([b[:5] for kaesten in d["bilder"].values() for b in kaesten if b[0] >= SCHWELLE], float)
    laenge = np.maximum(k[:, 3] - k[:, 1], k[:, 4] - k[:, 2])   # längere Seite des Kastens
    print(f"{sid}: Median {np.median(laenge):.1f} px aus {len(k)} Kästen (Sicherheit >= {SCHWELLE})")
