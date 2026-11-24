# BeeReal Paper 1, Versuche 02 und 03 (Filtervarianten, Bewertungsordner).
"""Filter-Varianten für SAM-3-Masken lokal durchrechnen (ohne GPU).

Eingabe: entpackte `rohdaten.zip` aus dem Kaggle-Notebook `beereal-sam3-filter` (bretter/*.png, masken.npz, sam3_roh.csv).
Ausgabe: Messwerte pro Biene, Trichter je Variante, Einzelfilter-Tabelle, Stichproben-Bögen, alle Bilder in Ordnern
und ein Bewertungsordner für die Handprüfung.

Aufruf:  python filter_auswertung.py <rohdaten-ordner> <ausgabe-ordner>
Danach optional (nach Handbewertung):  python filter_auswertung.py <rohdaten> <ausgabe> --optimieren

Schwellen sind relativ zum typischen Wert des jeweiligen Stocks formuliert (Median aller sicheren Masken), wo der
absolute Wert von Kamera-Abstand und Bildqualität abhängt (Größe, Schärfe). Damit gelten dieselben Zahlen für jeden Stock.
"""
import itertools, json, os, random, shutil, sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAW, OUT = sys.argv[1], sys.argv[2]
OPT = "--optimieren" in sys.argv
# --n_bewerten=150: Masken je Stock im Bewertungsordner; --regel=<optimierung.csv>: beste Regel je Stock anwenden (Spalte regel_best)
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
N_BEW = int(ARGS.get("n_bewerten", 150))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Daten laden
df = pd.read_csv(os.path.join(RAW, "sam3_roh.csv"))
npz = np.load(os.path.join(RAW, "masken.npz"))
BOARDS = {k: np.array(Image.open(os.path.join(RAW, "bretter", k + ".png")).convert("RGB")) for k in df.key.unique()}
MASK = {}            # (key, i) -> (y0, x0, Teilmaske bool)
for name in npz.files:
    key, i, y0, x0, h, w = name.split("|")
    h, w = int(h), int(w)
    MASK[(key, int(i))] = (int(y0), int(x0), np.unpackbits(npz[name])[: h * w].reshape(h, w).astype(bool))
df = df[[(k, i) in MASK for k, i in zip(df.key, df.i)]].reset_index(drop=True)
print("Masken:", len(df), "Bilder:", len(BOARDS))

def full_mask(key, i):
    y0, x0, m = MASK[(key, i)]
    f = np.zeros(BOARDS[key].shape[:2], bool); f[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return f

# ---------------------------------------------------------------- Messwerte pro Maske
def shape_stats(m):
    ys, xs = np.nonzero(m)
    n_lab, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    parts = int((st[1:, cv2.CC_STAT_AREA] >= 0.1 * m.sum()).sum())
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 0
    pts = np.stack([xs, ys], 1).astype(float); c = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts - c).T)) if len(pts) > 2 else (np.array([1.0, 1.0]), np.eye(2))
    v = evec[:, 1]
    return dict(parts=parts, solidity=float(m.sum() / max(hull, 1)), elong=float(np.sqrt(ev[1] / max(ev[0], 1e-6))),
                cx=float(c[0]), cy=float(c[1]), angle=float(np.degrees(np.arctan2(v[1], v[0]))), length=float(4 * np.sqrt(ev[1])))

def aligned_crop(board, m, st, out=96):
    """Längsachse senkrecht, größerer Teil (vermutlich Hinterleib) unten, Hintergrund schwarz."""
    side = int(max(st["length"] * 1.3, 16)); best = None
    for ang in (st["angle"] - 90, st["angle"] + 90):
        M = cv2.getRotationMatrix2D((st["cx"], st["cy"]), ang, 1.0); M[:, 2] += [side / 2 - st["cx"], side / 2 - st["cy"]]
        mm = cv2.warpAffine(m.astype(np.uint8), M, (side, side), flags=cv2.INTER_NEAREST)
        xs = np.nonzero(mm)[1]; w = (xs.max() - xs.min()) if len(xs) else 1e9
        if best is None or w < best[0]:
            best = (w, M, mm)
    _, M, mm = best
    im = cv2.warpAffine(board, M, (side, side), flags=cv2.INTER_LINEAR)
    if mm[: side // 2].sum() > mm[side // 2:].sum():
        im, mm = im[::-1, ::-1], mm[::-1, ::-1]
    im = np.ascontiguousarray(im * mm[..., None]); mm = np.ascontiguousarray(mm)
    return cv2.resize(im, (out, out), interpolation=cv2.INTER_AREA), cv2.resize(mm, (out, out), interpolation=cv2.INTER_NEAREST)

def sharpness(board, m):
    """Laplace-Varianz im Inneren der Maske, im Originalbild (nicht hochskaliert)."""
    g = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
    return float(lap[inner > 0].var()) if inner.sum() > 10 else 0.0

rows, ALIGNED, FULL = [], {}, {}
for r in df.itertuples():
    m = full_mask(r.key, r.i); b = BOARDS[r.key]
    st = shape_stats(m)
    # Wie lang ist die Kante, an der die Maske den Bildrand berührt, relativ zur Bienenlänge?
    edge = int(m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum())
    border = edge / max(st["length"], 1)
    ALIGNED[(r.key, r.i)] = aligned_crop(b, m, st)[0]
    rows.append(dict(parts=st["parts"], solidity=st["solidity"], elong=st["elong"], border=border, sharp=sharpness(b, m)))
df = pd.concat([df.drop(columns=[c for c in ("parts", "solidity", "elong", "border", "sharp") if c in df], errors="ignore"),
                pd.DataFrame(rows)], axis=1)
# Typische Werte je Stock aus den sicheren Masken (Sicherheit >= 0,5): Grundlage der relativen Schwellen
ref = df[df.score >= 0.5].groupby("hive")[["area", "sharp", "elong", "solidity"]].median()
df["size_rel"] = df.area / df.hive.map(ref.area)
df["sharp_rel"] = df.sharp / df.hive.map(ref.sharp)
df["elong_rel"] = df.elong / df.hive.map(ref.elong)
df["solidity_rel"] = df.solidity / df.hive.map(ref.solidity)
print("Typische Werte je Stock:\n", ref.round(1))

# Paarweise Überdeckung im selben Bild: Anteil der eigenen Fläche, den eine andere Maske überdeckt
PAIRS = {}       # (key, i) -> list of (j, Anteil)
for key, g in df.groupby("key"):
    ids = list(g.i); boxes = g[["x1", "y1", "x2", "y2"]].values
    for a, b_ in itertools.combinations(range(len(ids)), 2):
        A, B = boxes[a], boxes[b_]
        if A[0] > B[2] or B[0] > A[2] or A[1] > B[3] or B[1] > A[3]:
            continue
        ma, mb = full_mask(key, ids[a]), full_mask(key, ids[b_])
        inter = (ma & mb).sum()
        if inter:
            PAIRS.setdefault((key, ids[a]), []).append((ids[b_], inter / ma.sum()))
            PAIRS.setdefault((key, ids[b_]), []).append((ids[a], inter / mb.sum()))

# ---------------------------------------------------------------- Filter und Varianten
PRESETS = {
    "locker": dict(score=0.4, border=0.5, size_lo=0.5, size_hi=2.0, elong=0.65, solidity=0.85, overlap=0.30, sharp_rel=0.3),
    "mittel": dict(score=0.5, border=0.3, size_lo=0.5, size_hi=1.8, elong=0.75, solidity=0.90, overlap=0.20, sharp_rel=0.4),
    "streng": dict(score=0.6, border=0.15, size_lo=0.6, size_hi=1.5, elong=0.85, solidity=0.95, overlap=0.10, sharp_rel=0.5),
}
STEPS = [
    ("1_unsicher", "SAM-3-Sicherheit < score", lambda d, p: d.score < p["score"]),
    ("2_maske_kaputt", "Maske in mehrere Teile zerfallen", lambda d, p: d.parts > 1),
    ("3_am_rand", "am Bildrand abgeschnitten: Berührkante > border × Bienenlänge", lambda d, p: d.border > p["border"]),
    ("4_groesse", "Fläche relativ zur typischen Biene außerhalb [size_lo, size_hi]",
     lambda d, p: (d.size_rel < p["size_lo"]) | (d.size_rel > p["size_hi"])),
    ("5_form", "Längenverhältnis < elong × typisch oder Kompaktheit < solidity × typisch (von vorn, verdeckt, Klumpen)",
     lambda d, p: (d.elong_rel < p["elong"]) | (d.solidity_rel < p["solidity"])),
    ("6_ueberlappung", "von einer verbliebenen anderen Maske zu mehr als overlap überdeckt", None),
    ("7_unscharf", "Schärfe < sharp_rel × typische Schärfe des Stocks (nur Ausreißer, kein fester Anteil)",
     lambda d, p: d.sharp_rel < p["sharp_rel"]),
]

def apply(p):
    res = pd.Series("behalten", index=df.index)
    alive = pd.Series(True, index=df.index)
    for name, _, fn in STEPS:
        if name == "6_ueberlappung":
            alive_keys = {(k, i) for k, i in zip(df.key[alive], df.i[alive])}
            hit = pd.Series([alive[ix] and any(f > p["overlap"] and (k, j) in alive_keys for j, f in PAIRS.get((k, i), []))
                             for ix, k, i in zip(df.index, df.key, df.i)], index=df.index)
        else:
            hit = fn(df, p) & alive
        res[hit] = name; alive &= ~hit
    return res

def funnel(res, variant):
    out = []
    for hive in ["gelb", "rot", "alle"]:
        sel = (df.hive == hive) if hive != "alle" else pd.Series(True, index=df.index)
        n = int(sel.sum()); left = n
        out.append(dict(variante=variant, stock=hive, schritt="0_sam3_gefunden", weg=0, weg_prozent=0.0, uebrig=n, uebrig_prozent=100.0))
        for name, _, _ in STEPS:
            k = int((res[sel] == name).sum()); left -= k
            out.append(dict(variante=variant, stock=hive, schritt=name, weg=k, weg_prozent=round(100 * k / n, 1),
                            uebrig=left, uebrig_prozent=round(100 * left / n, 1)))
    return out

RESULTS = {v: apply(p) for v, p in PRESETS.items()}
for v, r in RESULTS.items():
    df["ergebnis_" + v] = r
FUN = pd.DataFrame([row for v, r in RESULTS.items() for row in funnel(r, v)])
FUN.to_csv(os.path.join(OUT, "trichter_varianten.csv"), index=False)

# Einzelfilter: Wie viel würde jeder Filter mit verschiedenen Werten allein entfernen?
single = []
for hive, g in df.groupby("hive"):
    for v in [0.3, 0.4, 0.5, 0.6, 0.7]: single.append(dict(stock=hive, filter="Sicherheit <", wert=v, weg_prozent=round(100 * (g.score < v).mean(), 1)))
    for v in [1.5, 1.7, 2.0, 2.5]: single.append(dict(stock=hive, filter="Größe > x·typisch", wert=v, weg_prozent=round(100 * (g.size_rel > v).mean(), 1)))
    for v in [0.4, 0.5, 0.6]: single.append(dict(stock=hive, filter="Größe < x·typisch", wert=v, weg_prozent=round(100 * (g.size_rel < v).mean(), 1)))
    for v in [0.15, 0.3, 0.5]: single.append(dict(stock=hive, filter="Randkante > x·Länge", wert=v, weg_prozent=round(100 * (g.border > v).mean(), 1)))
    for v in [0.65, 0.75, 0.85]: single.append(dict(stock=hive, filter="Längenverhältnis < x·typisch", wert=v, weg_prozent=round(100 * (g.elong_rel < v).mean(), 1)))
    for v in [0.85, 0.90, 0.95]: single.append(dict(stock=hive, filter="Kompaktheit < x·typisch", wert=v, weg_prozent=round(100 * (g.solidity_rel < v).mean(), 1)))
    for v in [0.3, 0.4, 0.5, 0.6, 0.7]: single.append(dict(stock=hive, filter="Schärfe < x·typisch", wert=v, weg_prozent=round(100 * (g.sharp_rel < v).mean(), 1)))
pd.DataFrame(single).to_csv(os.path.join(OUT, "einzelfilter.csv"), index=False)
if "regel" in ARGS:
    # Beste Einstellung je Stock aus einer früheren Optimierung (z. B. für den Blindtest auf neuen Masken)
    OB = pd.read_csv(ARGS["regel"]); df["regel_best"] = ""
    for hive in ["gelb", "rot"]:
        top = OB[OB.stock == hive].iloc[0]
        p = {k: top[k] for k in ["score", "border", "size_lo", "size_hi", "elong", "solidity", "overlap", "sharp_rel"]}
        print(f"Regel {hive}:", p)
        sel = df.hive == hive; df.loc[sel, "regel_best"] = apply(p)[sel]
df.drop(columns=[]).to_csv(os.path.join(OUT, "bienen_messwerte.csv"), index=False)

# ---------------------------------------------------------------- Bilder
def raw_with_outline(r, pad=0.4):
    b = BOARDS[r.key]; m = full_mask(r.key, r.i)
    w, h = r.x2 - r.x1, r.y2 - r.y1
    x1, y1 = int(max(r.x1 - pad * w, 0)), int(max(r.y1 - pad * h, 0))
    x2, y2 = int(min(r.x2 + pad * w, b.shape[1])), int(min(r.y2 + pad * h, b.shape[0]))
    im = np.ascontiguousarray(b[y1:y2, x1:x2])
    cnts, _ = cv2.findContours(np.ascontiguousarray(m[y1:y2, x1:x2]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnts, -1, (255, 0, 255), 1)
    return im

def picture(r, res):
    return ALIGNED[(r.key, r.i)] if res == "behalten" else raw_with_outline(r)

def sheet(sub, col, title, path, compare=None, n=48, cols=12):
    """Oben 2 Zeilen behaltene Bienen zum Vergleich (grün beschriftet), darunter die Stichprobe.
    Alle als Rohbild mit Umriss, damit man gleich aussehende Bilder vergleicht."""
    if sub.empty:
        return
    sub = sub.sample(min(n, len(sub)), random_state=0)
    top = compare.sample(min(2 * cols, len(compare)), random_state=2) if compare is not None and len(compare) else sub.iloc[:0]
    rows_ = int(np.ceil(len(top) / cols)) + int(np.ceil(len(sub) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 1.3, rows_ * 1.5), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    cells = list(axes.flat)
    k = 0
    for r in top.itertuples():
        ax = cells[k]; k += 1; ax.imshow(raw_with_outline(r))
        ax.set_title("VERGLEICH behalten", fontsize=6, color="green")
    k = int(np.ceil(len(top) / cols)) * cols
    for r in sub.itertuples():
        ax = cells[k]; k += 1; ax.imshow(raw_with_outline(r))
        ax.set_title(f"s{r.score:.2f} R{r.border:.2f} g{r.size_rel:.1f}\ne{r.elong_rel:.2f} k{r.solidity_rel:.2f} sch{r.sharp_rel:.1f}", fontsize=6)
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(path, dpi=85); plt.close()

for v in PRESETS:
    d = os.path.join(OUT, "boegen", v); os.makedirs(d, exist_ok=True)
    for hive in ["gelb", "rot"]:
        for name in ["behalten"] + [s[0] for s in STEPS]:
            sub = df[(df.hive == hive) & (df["ergebnis_" + v] == name)]
            keep = df[(df.hive == hive) & (df["ergebnis_" + v] == "behalten")]
            sheet(sub, "ergebnis_" + v, f"{v} | {hive}: {name} ({len(sub)} Bienen, Stichprobe)", os.path.join(d, f"{hive}_{name}.png"),
                  compare=None if name == "behalten" else keep)
        # behaltene zusätzlich gedreht und freigestellt (so gehen sie ins Training)
        keep = df[(df.hive == hive) & (df["ergebnis_" + v] == "behalten")].sample(frac=1, random_state=0).head(60)
        if len(keep):
            fig, axes = plt.subplots(5, 12, figsize=(15.6, 7.5), squeeze=False)
            for ax in axes.flat: ax.axis("off")
            for ax, r in zip(axes.flat, keep.itertuples()): ax.imshow(ALIGNED[(r.key, r.i)])
            fig.suptitle(f"{v} | {hive}: behalten, gedreht und freigestellt", fontsize=10); plt.tight_layout()
            plt.savefig(os.path.join(d, f"{hive}_behalten_gedreht.png"), dpi=85); plt.close()

# Übersicht je Variante: ein Brett pro Video, grün = behalten, rot = aussortiert (+ Filternummer)
for v in PRESETS:
    d = os.path.join(OUT, "uebersicht", v); os.makedirs(d, exist_ok=True)
    for key in [k for k in BOARDS if k.endswith("_f0600")]:
        im = BOARDS[key].copy()
        for r in df[df.key == key].itertuples():
            res = getattr(r, "ergebnis_" + v)
            cnts, _ = cv2.findContours(full_mask(key, r.i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = (0, 255, 0) if res == "behalten" else (255, 0, 0)
            cv2.drawContours(im, cnts, -1, c, 2)
            if res != "behalten":
                cv2.putText(im, res[0], (int(r.x1), int(r.y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        Image.fromarray(im).save(os.path.join(d, f"{key}.png"))

# Alle Einzelbilder, sortiert nach Ergebnis der Variante "mittel" (JPEG, damit das Repo klein bleibt)
IMG = os.path.join(OUT, "bilder_mittel"); shutil.rmtree(IMG, ignore_errors=True)
for r in df.itertuples():
    res = r.ergebnis_mittel; d = os.path.join(IMG, r.hive, res); os.makedirs(d, exist_ok=True)
    Image.fromarray(picture(r, res)).save(os.path.join(d, f"{r.key}_b{r.i:03d}.jpg"), quality=90)

# Bewertungsordner: 300 zufällige Masken (gleich verteilt auf beide Stöcke), immer als Rohbild mit Umriss.
# Jan verschiebt schlechte Bilder nach bewerten/schlecht/. Daraus werden später die besten Schwellen berechnet.
BEW = os.path.join(OUT, "bewerten")
if not os.path.exists(BEW):
    os.makedirs(os.path.join(BEW, "schlecht"))
    pick = pd.concat([g.sample(min(N_BEW, len(g)), random_state=1) for _, g in df.groupby("hive")])
    for r in pick.itertuples():
        Image.fromarray(raw_with_outline(r)).resize((128, 128), Image.NEAREST).save(
            os.path.join(BEW, f"{r.hive}_{r.key}_b{r.i:03d}.png"))

# ---------------------------------------------------------------- Optimieren nach Handbewertung
if OPT:
    # Ordner je Stock: <stock>/gut, <stock>/sehr_gut = gut; <stock>/schlecht/<grund> = schlecht;
    # direkt in <stock>/ = noch nicht bewertet
    lab, grund = {}, {}
    for root, _, files in os.walk(BEW):
        parts = os.path.relpath(root, BEW).replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        for f in files:
            if not f.endswith(".png"):
                continue
            n = f[:-4]
            if parts[1] == "schlecht":
                lab[n] = False; grund[n] = parts[2] if len(parts) > 2 else "ohne_grund"
            elif parts[1] in ("gut", "sehr_gut"):
                lab[n] = True; grund[n] = parts[1]
    print("Bewertet:", len(lab), "davon schlecht:", sum(not v for v in lab.values()))
    print(pd.Series(grund).value_counts().to_string())
    df["name"] = [f"{h}_{k}_b{i:03d}" for h, k, i in zip(df.hive, df.key, df.i)]
    L = df[df.name.isin(lab)].copy(); L["gut"] = L.name.map(lab)
    L["grund"] = L.name.map(grund)

    # 1) Ohne jeden Filter: wie viel ist überhaupt gut?
    print("\nOhne Filter, Anteil gut:", L.groupby("hive").gut.mean().round(3).to_dict())

    # 2) Trennt ein Messwert überhaupt "gut" von einem Grund? AUC: 0,5 = Zufall, 1,0 = perfekt.
    #    Wert < 0,5 heißt: der Fehler hat eher HÖHERE Werte.
    def auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        a = np.asarray(pos)[:, None]; b = np.asarray(neg)[None, :]
        return float(((a > b).mean() + 0.5 * (a == b).mean()))
    rows = []
    for hive in ["gelb", "rot"]:
        H = L[L.hive == hive]; good = H[H.gut]
        for g in sorted(H[~H.gut].grund.unique()):
            bad = H[H.grund == g]
            rows.append(dict(stock=hive, grund=g, n=len(bad), **{m: round(auc(good[m], bad[m]), 2) for m in
                        ["score", "border", "size_rel", "elong_rel", "solidity_rel", "sharp_rel", "parts"]}))
    TR = pd.DataFrame(rows); TR.to_csv(os.path.join(OUT, "trennschaerfe.csv"), index=False)
    print("\nTrennschärfe (AUC gut vs. Grund):\n" + TR.to_string(index=False))

    # 3) Welcher Filter fängt welchen Grund (je Preset)?
    for v, p in PRESETS.items():
        L["filter_" + v] = apply(p)[L.index].values
    for v in PRESETS:
        ct = pd.crosstab([L.hive, L.grund], L["filter_" + v])
        ct.to_csv(os.path.join(OUT, f"grund_vs_filter_{v}.csv"))
    print("\nGrund vs. Filter (mittel):\n" + pd.crosstab([L.hive, L.grund], L["filter_mittel"]).to_string())

    grid = dict(score=[0.4, 0.5, 0.6], border=[0.15, 0.3, 0.5], size_lo=[0.5, 0.6], size_hi=[1.5, 1.8, 2.0],
                elong=[0.65, 0.75, 0.85], solidity=[0.85, 0.9, 0.95], overlap=[0.1, 0.2, 0.3],
                sharp_rel=[0.3, 0.5, 0.7, 0.85, 1.0])
    best = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals)); res = apply(p)
        kept = res[L.index] == "behalten"
        for hive in ["gelb", "rot", "alle"]:
            s = (L.hive == hive) if hive != "alle" else pd.Series(True, index=L.index)
            tp = (kept & L.gut & s).sum(); fp = (kept & ~L.gut & s).sum(); fn = (~kept & L.gut & s).sum()
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f05 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)   # F0,5: Genauigkeit zählt doppelt ("lieber nichts als falsch")
            best.append(dict(stock=hive, **p, genauigkeit=round(prec, 3), ausbeute=round(rec, 3), f05=round(f05, 3)))
    B = pd.DataFrame(best).sort_values("f05", ascending=False)
    B.to_csv(os.path.join(OUT, "optimierung.csv"), index=False)
    print(B.groupby("stock").head(3).to_string(index=False))

    # 4) Beste Kombination je Stock: was fängt sie, was lässt sie durch?
    for hive in ["gelb", "rot"]:
        top = B[B.stock == hive].iloc[0]
        p = {k: top[k] for k in grid}
        H = L[L.hive == hive].copy(); H["filter"] = apply(p)[H.index].values
        ct = pd.crosstab(H.grund, H["filter"]); ct.to_csv(os.path.join(OUT, f"grund_vs_filter_best_{hive}.csv"))
        print(f"\nBeste für {hive}: {p}\n" + ct.to_string())

print(FUN[FUN.stock == "alle"].to_string(index=False))
print("FERTIG")
