# BeeReal Paper 1, Versuch 06 (Kennzahlen Einzelbild gegen Videomodus).
"""Lauf A (P2/P4): SAM 3 Video gegen SAM 3 Einzelbild + Positions-Tracker gegen Grounding DINO + Positions-Tracker.

Aufruf: python p2_auswertung.py <beereal_baseline.py> <out_p2-ordner> <gd-ordner mit dets_<sid>.json> <ausgabe-ordner>
Alle Längen in Sekunden; 10 Bilder/s (jedes 6. Videobild). Tracks aus dem Videomodus = Objekt-IDs von SAM 3.
"""
import gzip, json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

src = open(sys.argv[1], encoding="utf-8").read(); P2, GD, OUT = sys.argv[2], sys.argv[3], sys.argv[4]; os.makedirs(OUT, exist_ok=True)
ns = {"np": np, "linear_sum_assignment": linear_sum_assignment, "MAX_GAP": 5, "REL_DIST": 0.5}
exec(src[src.index("def track("):src.index("def remove_spikes")], ns); track = ns["track"]
cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); size = lambda b: max(b[2] - b[0], b[3] - b[1])
FPS = 59.94

def kennzahlen(tracks, frames, name, sid, clip, pro_bild):
    """tracks: Liste von Listen (frame, box). Kennzahlen wie im Bildraten-Test."""
    span = np.array([(t[-1][0] - t[0][0]) / FPS for t in tracks]) if tracks else np.array([0.0])
    step = frames[1] - frames[0]
    mitte = [t for t in tracks if t[-1][0] < frames[-1] - 2 * step]
    starts = [t for t in tracks if t[0][0] > frames[0]]; nah = []
    for t in mitte:
        fe, be = t[-1]; best = None
        for s in starts:
            fs, bs = s[0]
            if 0 < fs - fe <= 6 * FPS and np.linalg.norm(cen(bs) - cen(be)) / size(be) < 1.0 and (best is None or fs - fe < best):
                best = fs - fe
        if best is not None:
            nah.append(best / FPS)
    nah = np.array(nah)
    return dict(clip=f"{sid} {clip}", verfahren=name, bienen_pro_bild=round(pro_bild, 1), tracks=len(tracks),
                median_spanne_s=round(float(np.median(span)), 2), ab_2s=int((span >= 2).sum()), ab_5s=int((span >= 5).sum()),
                ab_9s=int((span >= 9).sum()), ab_20s=int((span >= 20).sum()),
                enden_mit_neuem_track_nah_prozent=round(100 * len(nah) / max(len(mitte), 1)),
                luecke_median_s=round(float(np.median(nah)), 2) if len(nah) else None)

rows = []
for sid in ("rec2123_seq065", "rec2125_seq012"):
    gd_all = {int(k): v for k, v in json.load(open(os.path.join(GD, f"dets_{sid}.json"))).items()}
    for clip, last in (("10s", 600), ("30s", 1800)):
        vp = os.path.join(P2, f"video_{sid}_{clip}.json")
        frames = [f for f in sorted(gd_all) if f < last]
        gd = {f: gd_all[f] for f in frames}
        rows.append(kennzahlen(list(track(gd).values()), frames, "Grounding DINO + Positions-Tracker", sid, clip, np.mean([len(v) for v in gd.values()])))
        bp = os.path.join(P2, f"bild_{sid}_{clip}.json")
        if os.path.exists(bp):
            b = {int(k): v for k, v in json.load(open(bp)).items()}
            for thr in (0.3, 0.5):
                d = {f: [o[1:] for o in b[f] if o[0] >= thr] for f in sorted(b)}
                rows.append(kennzahlen(list(track(d).values()), sorted(d), f"SAM 3 Einzelbild (≥{thr}) + Positions-Tracker", sid, clip, np.mean([len(v) for v in d.values()])))
        if os.path.exists(vp):
            v = {int(k): o for k, o in json.load(open(vp)).items()}
            T = {}
            for f in sorted(v):
                for o in v[f]:
                    T.setdefault(o[0], []).append((f, o[2:]))
            rows.append(kennzahlen(list(T.values()), sorted(v), "SAM 3 Video (eigener Tracker)", sid, clip, np.mean([len(x) for x in v.values()])))
        print(*rows[-3:], sep="\n", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "p2_ergebnis.csv"), index=False); print(R.to_string(index=False))
