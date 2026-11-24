# BeeReal Paper 1, Versuch 10.
"""Baut ein Labeling-Paket für CVAT: Bilder (ZIP) + Vorschläge mit Track-IDs im Format MOT 1.1 (ZIP).

Die Vorschläge kommen aus SAM-3-Einzelbild-Kästen (Datei bild_<sid>_*.json: {bild: [[sicherheit, x1, y1, x2, y2], ...]},
Koordinaten im ROI-Ausschnitt) und dem ByteTrack-artigen Tracker (hoch >= 0,7 startet Tracks, tief >= 0,3 setzt nur fort).

Aufruf: python cvat_paket.py <video.mp4> <bild_json oder - für ohne Vorschläge> <gelb|rot> <ausgabeordner> [start_s=0] [dauer_s=10] [bilder_pro_s=5]
Ergebnis: <name>_bilder.zip (in CVAT als neue Aufgabe hochladen) und <name>_vorschlaege_mot.zip (Upload annotations, MOT 1.1).
"""
import io, json, os, sys, zipfile
import cv2, numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

video, bild_p, hive, out = sys.argv[1:5]
START = float(sys.argv[5]) if len(sys.argv) > 5 else 0
DAUER = float(sys.argv[6]) if len(sys.argv) > 6 else 10
BPS = float(sys.argv[7]) if len(sys.argv) > 7 else 5
ROI = {"gelb": (90, 800, 1080, 1025), "rot": (0, 750, 1080, 1035)}[hive]
FPS = 59.94; HOCH, TIEF, MAX_GAP, REL = 0.7, 0.3, 5, 0.5
sid = os.path.basename(video).replace(".mp4", "")
name = f"{sid}_{int(START)}-{int(START + DAUER)}s"
os.makedirs(out, exist_ok=True)

schritt = max(1, round(FPS / BPS))
wunsch = list(range(int(START * FPS), int((START + DAUER) * FPS), schritt))
if bild_p == "-":  # ohne Vorschläge (z. B. Test, bevor Detektionen da sind)
    alle = {f: [] for f in wunsch}
else:
    alle = {int(k): v for k, v in json.load(open(bild_p)).items()}
frames = sorted(set(min(alle, key=lambda f: abs(f - w)) for w in wunsch))  # nächstes Bild mit Kästen

cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); size = lambda b: max(b[2] - b[0], b[3] - b[1])
def match(tr, dets):
    if not tr or not dets: return [], list(range(len(tr))), list(range(len(dets)))
    C = np.array([[np.linalg.norm(cen(t[-1][1]) - cen(d)) / size(t[-1][1]) for d in dets] for t in tr])
    r, c = linear_sum_assignment(C); ok = [(i, j) for i, j in zip(r, c) if C[i, j] < REL]
    return ok, [i for i in range(len(tr)) if i not in {a for a, _ in ok}], [j for j in range(len(dets)) if j not in {b for _, b in ok}]

tracks = []
for k, f in enumerate(frames):
    akt = [t for t in tracks if k - t[-1][0] <= MAX_GAP]
    H = [o[1:] for o in alle[f] if o[0] >= HOCH]; L = [o[1:] for o in alle[f] if TIEF <= o[0] < HOCH]
    ok, rest, neu = match(akt, H)
    for i, j in ok: akt[i].append((k, H[j]))
    ok2, _, _ = match([akt[i] for i in rest], L)
    for a, j in ok2: akt[rest[a]].append((k, L[j]))
    for j in neu: tracks.append([(k, H[j])])

# Bilder
cap = cv2.VideoCapture(video); zb = zipfile.ZipFile(os.path.join(out, f"{name}_bilder.zip"), "w")
for k, f in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    buf = io.BytesIO(); Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).crop(ROI).save(buf, "JPEG", quality=92)
    zb.writestr(f"{k + 1:06d}.jpg", buf.getvalue())
zb.close()

# Vorschläge im MOT-1.1-Format: frame(ab 1), id, x, y, w, h, nicht_ignorieren, klasse, sichtbarkeit
zeilen = []
for tid, t in enumerate(tracks, 1):
    for k, b in t:
        zeilen.append(f"{k + 1},{tid},{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},1,1,1.0")
zeilen.sort(key=lambda z: (int(z.split(",")[0]), int(z.split(",")[1])))
if zeilen:
  with zipfile.ZipFile(os.path.join(out, f"{name}_vorschlaege_mot.zip"), "w") as za:
    za.writestr("gt/gt.txt", "\n".join(zeilen) + "\n"); za.writestr("gt/labels.txt", "bee\n")
json.dump(dict(video=sid, start_s=START, dauer_s=DAUER, bilder_pro_s=BPS, videobilder=frames, roi=ROI,
               tracker=dict(hoch=HOCH, tief=TIEF, max_luecke=MAX_GAP, abstand=REL)),
          open(os.path.join(out, f"{name}_info.json"), "w"), indent=1)
print(name, "Bilder:", len(frames), "Tracks:", len(tracks), "Kästen:", len(zeilen))
