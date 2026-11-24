# BeeReal Paper 1, Versuch 04.
"""Tracks nachträglich zusammenkleben (lokal, ohne GPU). Idee: StrongSORT++ (AFLink + GSI), Bozek et al. 2021.

Der Positions-Tracker liefert kurze, saubere Track-Stücke. Weil die Biene oft nur 1-3 s vom Detektor nicht gefunden wird,
reißen Tracks ab. Hier wird offline entschieden, welches Ende zu welchem Anfang gehört:
  * Kandidat nur, wenn der neue Track höchstens MAX_LUECKE Bilder nach dem Ende beginnt und weniger als MAX_DIST
    Bienenlängen entfernt liegt, ohne zeitliche Überschneidung.
  * Zuordnung als lineare Zuordnung (Hungarian), also jedes Ende höchstens einem Anfang.
  * Nur "eindeutig": der zweitbeste Kandidat von Ende UND Anfang muss mindestens EINDEUTIG-mal so weit weg sein
    (oder es gibt keinen).
  * Mehrere Runden, damit auch Ketten entstehen. Lücken werden linear aufgefüllt (GSI-Vorläufer).

Aufruf:  python tracks_kleben.py <beereal_baseline.py> <dets_*.json[.gz] ...> [--out=<ordner>] [--luecke=60 --dist=1.0 --eindeutig=2.0]
Ausgabe: <out>/kleben_statistik.csv, <out>/<sid>_geklebt.json (Tracks mit Kästen, "interpoliert" je Bild markiert)
"""
import gzip, json, os, re, sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
MAX_LUECKE = int(ARGS.get("luecke", 60))      # in Bildern à 0,1 s (60 = 6 s)
MAX_DIST = float(ARGS.get("dist", 1.0))       # in Bienenlängen
EINDEUTIG = float(ARGS.get("eindeutig", 2.0))  # zweitbester Kandidat mindestens so viel weiter weg
RUNDEN = int(ARGS.get("runden", 3))
RICHTUNG = ARGS.get("richtung", "0") == "1"      # Laufrichtung der Biene mitbenutzen (Position am Ende vorausrechnen)
OUT = ARGS.get("out", "kleben_ergebnis"); os.makedirs(OUT, exist_ok=True)

# Tracker aus dem Baseline-Skript übernehmen (identische Einstellungen wie in den Läufen)
src = open(POS[0], encoding="utf-8").read()
ns = {"np": np, "linear_sum_assignment": linear_sum_assignment, "MAX_GAP": 5, "REL_DIST": 0.5}
exec(src[src.index("def track("):src.index("def remove_spikes")], ns)
track = ns["track"]

cen = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
size = lambda b: max(b[2] - b[0], b[3] - b[1])


def tempo(t, am_ende, step, n=10):
    """Geschwindigkeit in Pixel pro Bild aus den letzten (am_ende) bzw. ersten n Kästen eines Track-Stücks."""
    w = t[-n:] if am_ende else t[:n]
    if len(w) < 3:
        return np.zeros(2)
    return (cen(w[-1][1]) - cen(w[0][1])) / max((w[-1][0] - w[0][0]) / step, 1)


def kleben(tracks, step):
    """tracks: Liste von Listen (frame, box). Gibt neue Liste und die Klebestellen zurück."""
    tracks = [list(t) for t in tracks]; stellen = []
    for _ in range(RUNDEN):
        n = len(tracks)
        ende = [(t[-1][0], cen(t[-1][1]), size(t[-1][1])) for t in tracks]
        anf = [(t[0][0], cen(t[0][1]), size(t[0][1])) for t in tracks]
        V_END = [tempo(t, True, step) for t in tracks]; V_ANF = [tempo(t, False, step) for t in tracks]
        INF = 1e6; C = np.full((n, n), INF)
        for a in range(n):
            fe, ce, se = ende[a]
            for b in range(n):
                if a == b:
                    continue
                fs, cs, ss = anf[b]
                gap = (fs - fe) // step
                if 0 < gap <= MAX_LUECKE:
                    d = np.linalg.norm(cs - ce) / min(se, ss)
                    if RICHTUNG and d < 2 * MAX_DIST:
                        # Wo müsste die Biene nach der Lücke sein, wenn sie weiterläuft? Und rückwärts vom neuen Anfang.
                        pa = ce + V_END[a] * gap; pb = cs - V_ANF[b] * gap
                        d = 0.5 * (np.linalg.norm(cs - pa) + np.linalg.norm(ce - pb)) / min(se, ss)
                    if d < MAX_DIST:
                        C[a, b] = d + 0.01 * gap          # kleine Zeitstrafe: näher in der Zeit ist besser
        if (C < INF).sum() == 0:
            break
        # Eindeutigkeit: zweitbester Kandidat je Ende (Zeile) und je Anfang (Spalte)
        def zweit(M):
            s = np.sort(M, axis=1); return s[:, 0], s[:, 1] if M.shape[1] > 1 else np.full(len(M), INF)
        r_best, r_second = zweit(C); c_best, c_second = zweit(C.T)
        rows, cols = linear_sum_assignment(C)
        merged, used = {}, set()
        for a, b in zip(rows, cols):
            if C[a, b] >= INF:
                continue
            ok_a = r_second[a] >= INF or r_second[a] >= EINDEUTIG * C[a, b]
            ok_b = c_second[b] >= INF or c_second[b] >= EINDEUTIG * C[a, b]
            if ok_a and ok_b:
                merged[a] = b; used.add(b)
        if not merged:
            break
        # Ketten bilden
        new, done = [], set()
        for a in range(n):
            if a in used or a in done:
                continue
            chain, cur = [a], a
            while cur in merged:
                cur = merged[cur]; chain.append(cur)
            done.update(chain)
            t = []
            for x, y in zip(chain, chain[1:] + [None]):
                t += tracks[x]
                if y is not None:
                    (f0, b0), (f1, b1) = tracks[x][-1], tracks[y][0]
                    stellen.append(dict(von=f0, nach=f1, box_von=[float(v) for v in b0], box_nach=[float(v) for v in b1], luecke_bilder=(f1 - f0) // step,
                                        dist=float(np.linalg.norm(cen(b1) - cen(b0)) / min(size(b0), size(b1)))))
                    for f in range(f0 + step, f1, step):          # lineare Interpolation (markiert als interpoliert)
                        w = (f - f0) / (f1 - f0)
                        t.append((f, [float(p + w * (q - p)) for p, q in zip(b0, b1)] + ["interpoliert"]))
            new.append(t)
        tracks = new
    return tracks, stellen


rows = []
for path in POS[1:]:
    sid = re.search(r"dets_(rec\d+_seq\d+)", path).group(1)
    d = {int(k): v for k, v in json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path)).items()}
    F = sorted(d); step = F[1] - F[0]
    vor = list(track(d).values())
    nach, stellen = kleben(vor, step)
    L0 = np.array([len(t) for t in vor]); L1 = np.array([sum(1 for _, b in t if len(b) == 4) for t in nach])
    S = pd.DataFrame(stellen)
    json.dump(stellen, open(os.path.join(OUT, f"{sid}_stellen.json"), "w"))
    lockerer = np.array([len(t) for t in track(d, 50, 1.0).values()])    # zum Vergleich: lockerer Tracker (Risiko Verwechslung)
    rows.append(dict(sid=sid, tracks_vorher=len(vor), tracks_nachher=len(nach), klebestellen=len(stellen),
                     median_vorher=float(np.median(L0)), median_nachher=float(np.median(L1)),
                     ab10_vorher=int((L0 >= 10).sum()), ab10_nachher=int((L1 >= 10).sum()),
                     ab100_vorher=int((L0 >= 100).sum()), ab100_nachher=int((L1 >= 100).sum()),
                     luecke_median_bilder=float(S.luecke_bilder.median()) if len(S) else 0,
                     dist_median=round(float(S.dist.median()), 2) if len(S) else 0,
                     locker_tracks=len(lockerer), locker_ab100=int((lockerer >= 100).sum())))
    json.dump([[(f, b) for f, b in t] for t in nach], open(os.path.join(OUT, f"{sid}_geklebt.json"), "w"))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "kleben_statistik.csv"), index=False)
print(R.to_string(index=False))
