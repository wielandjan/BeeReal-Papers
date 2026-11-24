# BeeReal Paper 1, Versuch 05.
"""Bildraten-Test (V6): Werden Tracks besser, wenn mehr Bilder pro Sekunde ausgewertet werden?

Eingabe: dets_<sid>_fps30.json.gz aus dem Kaggle-Notebook beereal_fps_test.py (erste 10 s, jedes 2. Bild).
Daraus werden 30, 15 und 10 Bilder/s abgeleitet (jedes 1., 2., 3. gespeicherte Bild). Der Tracker bekommt immer dieselbe
Lückenzeit von 0,5 s (MAX_GAP in Bildern wird mitskaliert), Abstand 0,5 Bienenlängen. Alles in Sekunden gemessen.

Aufruf: python fps_auswertung.py <beereal_baseline.py> <ausgabe-ordner> <dets_*_fps30.json.gz ...>
"""
import gzip, json, os, re, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

src = open(sys.argv[1], encoding="utf-8").read(); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
ns = {"np": np, "linear_sum_assignment": linear_sum_assignment, "MAX_GAP": 5, "REL_DIST": 0.5}
exec(src[src.index("def track("):src.index("def remove_spikes")], ns)
track = ns["track"]
cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); size = lambda b: max(b[2] - b[0], b[3] - b[1])
FPS_BASIS = 59.94 / 2          # gespeicherte Bilder pro Sekunde
rows, luecken = [], []
for path in sys.argv[3:]:
    sid = re.search(r"dets_(rec\d+_seq\d+)", path).group(1)
    d = {int(k): v for k, v in json.load(gzip.open(path, "rt")).items()}; F = sorted(d)
    for name, every in (("30 Bilder/s", 1), ("15 Bilder/s", 2), ("10 Bilder/s", 3)):
        fr = F[::every]; fps = FPS_BASIS / every; sub = {f: d[f] for f in fr}
        gap = max(int(round(0.5 * fps)), 1)
        T = list(track(sub, gap, 0.5).values())
        L = np.array([len(t) for t in T]) / fps                          # Länge in Sekunden (Anzahl Kästen / Bildrate)
        span = np.array([(t[-1][0] - t[0][0]) / 59.94 for t in T])        # Zeitspanne in Sekunden
        mitte = [t for t in T if t[-1][0] < fr[-1] - 2 * (fr[1] - fr[0])] # Enden mitten im Clip
        # Lücke bis zu einem neuen Track in der Nähe (< 1 Länge), in Sekunden
        starts = [t for t in T if t[0][0] > fr[0]]; res = []
        for t in mitte:
            fe, be = t[-1]; best = None
            for s in starts:
                fs, bs = s[0]
                if 0 < fs - fe <= 6 * 59.94:
                    dd = np.linalg.norm(cen(bs) - cen(be)) / size(be)
                    if dd < 1.0 and (best is None or fs - fe < best):
                        best = fs - fe
            res.append(None if best is None else best / 59.94)
        nah = np.array([r for r in res if r is not None])
        rows.append(dict(clip=sid, bildrate=name, bilder=len(fr), bienen_pro_bild=round(float(np.mean([len(v) for v in sub.values()])), 1),
                         tracks=len(T), median_spanne_s=round(float(np.median(span)), 2),
                         ab_2s=int((span >= 2).sum()), ab_5s=int((span >= 5).sum()), ab_9s=int((span >= 9).sum()),
                         enden_mitte=len(mitte), enden_mit_neuem_track_nah=round(100 * len(nah) / max(len(mitte), 1)),
                         luecke_median_s=round(float(np.median(nah)), 2) if len(nah) else None,
                         luecke_bis_0_5s_prozent=round(100 * float((nah <= 0.5).mean())) if len(nah) else None,
                         luecke_ueber_2s_prozent=round(100 * float((nah > 2).mean())) if len(nah) else None))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "bildrate_ergebnis.csv"), index=False); print(R.to_string(index=False))
