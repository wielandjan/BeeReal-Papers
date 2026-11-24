# BeeReal Paper 1, Versuch 04 (Prüfbögen).
"""Bögen zum Prüfen der Klebestellen: je Zeile links das Bild beim Ende des alten Tracks, rechts beim Anfang des neuen.
Grün = Kasten des alten Tracks am Ende, Rot = Kasten des neuen Tracks am Anfang (im linken Bild zusätzlich der rote Kasten
gestrichelt an seiner späteren Position). Frage an das Auge: Ist die Biene im rechten Bild dieselbe wie im linken?

Aufruf: python klebestellen_bogen.py <video.mp4> <stellen.json> <ausgabe.png> <gelb|rot> [anzahl=24] [seed=0]
"""
import json, random, sys
import cv2, numpy as np
from PIL import Image, ImageDraw

video, stellen_p, out, hive = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 24
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
ROI = {"gelb": (90, 800, 1080, 1025), "rot": (0, 750, 1080, 1035)}[hive]
st = json.load(open(stellen_p)); random.Random(SEED).shuffle(st); st = st[:N]
cap = cv2.VideoCapture(video)

def frame(f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).crop(ROI)

T = 170; cols = 3; rows = (len(st) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 2 * T + 8 * cols, rows * (T + 16)), "white"); d = ImageDraw.Draw(sheet)
for k, s in enumerate(st):
    a, b = s["box_von"], s["box_nach"]; size = max(a[2] - a[0], a[3] - a[1])
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2; half = max(size * 2.2, 40)
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    x0 = (k % cols) * (2 * T + 8); y0 = (k // cols) * (T + 16)
    for j, (f, col) in enumerate([(s["von"], "lime"), (s["nach"], "red")]):
        im = frame(f).crop(box).resize((T, T), Image.BICUBIC); dd = ImageDraw.Draw(im); sc = T / (2 * half)
        for bx, c, dash in ((a, "lime", j == 1), (b, "red", j == 0)):
            r = [(bx[0] - box[0]) * sc, (bx[1] - box[1]) * sc, (bx[2] - box[0]) * sc, (bx[3] - box[1]) * sc]
            dd.rectangle(r, outline=c, width=1 if dash else 2)
        sheet.paste(im, (x0 + j * T, y0 + 14))
    d.text((x0 + 2, y0 + 1), f"{k+1}: Lücke {s['luecke_bilder']/10:.1f} s, Abstand {s['dist']:.2f} Längen", fill="black")
sheet.save(out); print("OK", out, len(st))
