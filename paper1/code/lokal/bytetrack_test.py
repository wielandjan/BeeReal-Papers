# BeeReal Paper 1, Versuch 06 (ByteTrack-Prinzip gegen Positions-Tracker auf denselben SAM-3-Kästen).
"""ByteTrack-artiger Test (lokal, ohne GPU): niedrig sichere SAM-3-Detektionen verlängern nur bestehende Tracks, starten aber keine neuen.
Aufruf: python bytetrack_test.py <beereal_baseline.py> <ordner mit bild_*_10s.json> <ausgabe.csv>
"""
import json, os, sys, re
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
src = open(sys.argv[1], encoding="utf-8").read(); P2, OUT = sys.argv[2], sys.argv[3]
ns = {"np": np, "linear_sum_assignment": linear_sum_assignment, "MAX_GAP": 5, "REL_DIST": 0.5}
exec(src[src.index("def track("):src.index("def remove_spikes")], ns); track = ns["track"]
cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); size = lambda b: max(b[2] - b[0], b[3] - b[1])
FPS = 59.94; MAX_GAP = 5; REL = 0.5
p2 = open(os.path.join(os.path.dirname(__file__), "p2_auswertung.py"), encoding="utf-8").read()
exec(p2[p2.index("def kennzahlen"):p2.index("rows = []")], {"np": np, "FPS": FPS, "cen": cen, "size": size}, ns2 := {}); kennzahlen = ns2["kennzahlen"]

def match(tr, dets):
    """Zuordnung aktiver Tracks zu Kästen nach Abstand der Mitten / Größe. Gibt (paare, übrige Tracks, übrige Kästen)."""
    if not tr or not dets: return [], list(range(len(tr))), list(range(len(dets)))
    C = np.array([[np.linalg.norm(cen(t[-1][1]) - cen(d)) / size(t[-1][1]) for d in dets] for t in tr])
    r, c = linear_sum_assignment(C); ok = [(i, j) for i, j in zip(r, c) if C[i, j] < REL]
    return ok, [i for i in range(len(tr)) if i not in {a for a, _ in ok}], [j for j in range(len(dets)) if j not in {b for _, b in ok}]

def bytetrack(d, hoch, tief, stellen=None):
    frames = sorted(d); tracks = []; step = frames[1] - frames[0]
    for f in frames:
        akt = [t for t in tracks if f - t[-1][0] <= MAX_GAP * step]
        H = [o[1:] for o in d[f] if o[0] >= hoch]; L = [o[1:] for o in d[f] if tief <= o[0] < hoch]
        ok, rest, newH = match(akt, H)
        for i, j in ok: akt[i].append((f, H[j]))
        if tief < hoch:
            ok2, _, _ = match([akt[i] for i in rest], L)
            for a, j in ok2:
                t = akt[rest[a]]
                if stellen is not None:
                    stellen.append(dict(von=t[-1][0], nach=f, box_von=t[-1][1], box_nach=L[j], luecke_bilder=(f - t[-1][0]) // step,
                                        dist=float(np.linalg.norm(cen(t[-1][1]) - cen(L[j])) / size(t[-1][1]))))
                t.append((f, L[j]))
        for j in newH: tracks.append([(f, H[j])])
    return tracks

rows = []
for sid in ("rec2123_seq065", "rec2125_seq012"):
    b = {int(k): v for k, v in json.load(open(os.path.join(P2, f"bild_{sid}_10s.json"))).items()}; fr = sorted(b)
    pb = lambda thr: np.mean([sum(o[0] >= thr for o in b[f]) for f in fr])
    for thr in (0.3, 0.5):
        d = {f: [o[1:] for o in b[f] if o[0] >= thr] for f in fr}
        rows.append(kennzahlen(list(track(d).values()), fr, f"Basis: Positions-Tracker (≥{thr})", sid, "10s", pb(thr)))
    for hoch, tief in ((0.5, 0.3), (0.7, 0.3), (0.7, 0.5)):
        rows.append(kennzahlen(bytetrack(b, hoch, tief), fr, f"ByteTrack-artig (hoch ≥{hoch}, tief ≥{tief})", sid, "10s", pb(tief)))
    st = []; bytetrack(b, 0.7, 0.3, st)
    json.dump(st, open(os.path.join(os.path.dirname(OUT) or ".", f"bytetrack_stellen_{sid}.json"), "w")); print(sid, "Fortsetzungen mit unsicherem Kasten:", len(st))
R = pd.DataFrame(rows); R.to_csv(OUT, index=False); print(R.drop(columns=["clip"]).to_string(index=False))
