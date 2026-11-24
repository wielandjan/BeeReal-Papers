# BeeReal Paper 1, Versuch 08 (Tracks aller 69 Sequenzen nach dem ByteTrack-Prinzip).
"""Tracks aus den SAM-3-Kästen von Stufe B (ByteTrack-Prinzip), für alle Sequenzen eines Ordners.

Tracker: Kästen ab HOCH starten und setzen Tracks fort, Kästen ab TIEF setzen nur fort (zweite Zuordnung).
Zuordnung über Abstand der Mitten / Größe des letzten Kastens < REL, Lücke bis MAX_GAP Schritte. Danach remove_spikes wie Baseline.
Pro Kasten wird gespeichert: Bild, Kasten, Sicherheit, Fläche, Füllgrad, Überlappung (Maske), iou_max (größte IoU mit einem
anderen Kasten >= 0,5 im selben Bild) und rand (Kasten berührt den ROI-Rand).

Aufruf: python stufe_b_tracks.py <ordner mit sam3_*.json.gz> <ausgabeordner>
"""
import glob, gzip, json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

IN, OUT = sys.argv[1], sys.argv[2]
HOCH, TIEF, MAX_GAP, REL = 0.7, 0.3, 5, 0.5
os.makedirs(OUT, exist_ok=True)
cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); size = lambda b: max(b[2] - b[0], b[3] - b[1])

def match(tr, dets):
    if not tr or not dets:
        return [], list(range(len(tr))), list(range(len(dets)))
    pc = np.array([cen(t[-1]["b"]) for t in tr]); ps = np.array([size(t[-1]["b"]) for t in tr])
    dc = np.array([cen(d["b"]) for d in dets])
    C = np.linalg.norm(pc[:, None] - dc[None], axis=2) / ps[:, None]
    r, c = linear_sum_assignment(np.where(C < REL, C, 1e6))
    ok = [(i, j) for i, j in zip(r, c) if C[i, j] < REL]
    a, b = {i for i, _ in ok}, {j for _, j in ok}
    return ok, [i for i in range(len(tr)) if i not in a], [j for j in range(len(dets)) if j not in b]

def iou_max(boxes, stark):
    """größte IoU jedes Kastens mit einem anderen Kasten, dessen Sicherheit >= 0,5 ist"""
    B = np.array(boxes, float).reshape(-1, 4)
    if len(B) < 2:
        return np.zeros(len(B))
    x1 = np.maximum(B[:, None, 0], B[None, :, 0]); y1 = np.maximum(B[:, None, 1], B[None, :, 1])
    x2 = np.minimum(B[:, None, 2], B[None, :, 2]); y2 = np.minimum(B[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    iou = inter / (area[:, None] + area[None] - inter + 1e-9)
    np.fill_diagonal(iou, 0); iou[:, ~np.array(stark)] = 0
    return iou.max(1)

def remove_spikes(t):
    if len(t) < 3:
        return t
    c = np.array([cen(x["b"]) for x in t]); s = np.median([size(x["b"]) for x in t]) * 0.3
    keep = [True] * len(t)
    for i in range(1, len(t) - 1):
        if np.linalg.norm(c[i] - c[i - 1]) > s and np.linalg.norm(c[i] - c[i + 1]) > s and np.linalg.norm(c[i - 1] - c[i + 1]) < s:
            keep[i] = False
    return [x for x, k in zip(t, keep) if k]

stats = []
for p in sorted(glob.glob(os.path.join(IN, "sam3_*.json.gz"))):
    d = json.load(gzip.open(p, "rt")); sid = d["sid"]; roi = d["roi"]; W, H = roi[2] - roi[0], roi[3] - roi[1]
    frames = sorted(int(f) for f in d["bilder"]); step = d["step"]; tracks = []
    for f in frames:
        rows = d["bilder"][str(f)]
        boxes = [r[1:5] for r in rows]; im = iou_max(boxes, [r[0] >= 0.5 for r in rows])
        dets = [dict(f=f, b=[round(x, 1) for x in r[1:5]], s=r[0], a=r[5], fill=r[6], ov=r[7], iou=round(float(m), 3),
                     rand=int(r[1] <= 2 or r[2] <= 2 or r[3] >= W - 2 or r[4] >= H - 2)) for r, m in zip(rows, im)]
        akt = [t for t in tracks if (f - t[-1]["f"]) // step <= MAX_GAP]
        Hd = [x for x in dets if x["s"] >= HOCH]; Ld = [x for x in dets if TIEF <= x["s"] < HOCH]
        ok, rest, neu = match(akt, Hd)
        for i, j in ok: akt[i].append(Hd[j])
        ok2, _, _ = match([akt[i] for i in rest], Ld)
        for a, j in ok2: akt[rest[a]].append(Ld[j])
        for j in neu: tracks.append([Hd[j]])
    tracks = [remove_spikes(t) for t in tracks]
    out = [dict(id=k, first=t[0]["f"], last=t[-1]["f"], kaesten=[[x["f"], *x["b"], x["s"], x["a"], x["fill"], x["ov"], x["iou"], x["rand"]] for x in t])
           for k, t in enumerate(tracks)]
    felder = ["f", "x1", "y1", "x2", "y2", "score", "flaeche", "fuellgrad", "ueberlappung", "iou_max", "rand"]
    json.dump(dict(sid=sid, roi=roi, step=step, felder=felder, tracks=out), gzip.open(os.path.join(OUT, f"tracks_{sid}.json.gz"), "wt"))
    L = np.array([(t["last"] - t["first"]) / 59.94 for t in out])
    n10 = sum(len(t["kaesten"]) >= 10 for t in out)
    stats.append(dict(sid=sid, tracks=len(out), ab_10_kaesten=n10, median_s=round(float(np.median(L)), 2), ab_10s=int((L >= 10).sum())))
    print(stats[-1], flush=True)
import pandas as pd
pd.DataFrame(stats).to_csv(os.path.join(OUT, "tracks_statistik.csv"), index=False)
