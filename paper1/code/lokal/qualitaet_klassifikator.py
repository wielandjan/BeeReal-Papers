# BeeReal Paper 1, Versuch 02 (gelernter Filter).
"""Gelernter Qualitätsfilter für SAM-3-Masken (lokal, CPU reicht).

Idee: Jans Handbewertung (gut / schlecht) als Lehrer. Jede Maske wird als Bildausschnitt durch DINOv2-small
geschickt (dasselbe Grundmodell wie in der Re-ID). Auf den 384 Merkmalen lernt eine logistische Regression
"gut oder schlecht". Verglichen wird mit den Messwerten allein und mit beiden zusammen.
Ehrlich gemessen: Jede Bewertung wird nur von einem Modell vorhergesagt, das sie nicht gesehen hat (Kreuzvalidierung).

Aufruf:  python qualitaet_klassifikator.py <rohdaten-ordner> <varianten-ordner>
Ausgabe: <varianten-ordner>/klassifikator/
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

RAW, VAR = sys.argv[1], sys.argv[2]
OUT = os.path.join(VAR, "klassifikator"); os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count())

# ---------------------------------------------------------------- Daten und Bewertung
df = pd.read_csv(os.path.join(VAR, "bienen_messwerte.csv"))
df["name"] = [f"{h}_{k}_b{i:03d}" for h, k, i in zip(df.hive, df.key, df.i)]
lab, grund = {}, {}
BEW = os.path.join(VAR, "bewerten")
for root, _, files in os.walk(BEW):
    parts = os.path.relpath(root, BEW).replace("\\", "/").split("/")
    if len(parts) < 2:
        continue
    for f in files:
        if not f.endswith(".png") or " - Kopie" in f:      # Windows-Kopien nicht doppelt zählen
            continue
        n = f[:-4]
        if parts[1] == "schlecht":
            lab[n] = 0; grund[n] = parts[2] if len(parts) > 2 else "ohne_grund"
        elif parts[1] in ("gut", "sehr_gut"):
            lab[n] = 1; grund[n] = parts[1]
df["gut"] = df.name.map(lab); df["grund"] = df.name.map(grund)
L = df[df.gut.notna()].copy(); L["gut"] = L.gut.astype(int)
print("Bewertet:", len(L), L.groupby("hive").gut.mean().round(3).to_dict(), flush=True)

# ---------------------------------------------------------------- Merkmale
BOARDS = {}
def crop(r, pad=0.25, size=224):
    if r.key not in BOARDS:
        BOARDS[r.key] = Image.open(os.path.join(RAW, "bretter", r.key + ".png")).convert("RGB")
    b = BOARDS[r.key]; w, h = r.x2 - r.x1, r.y2 - r.y1; s = max(w, h) * (1 + 2 * pad)
    cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
    return b.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((size, size), Image.BICUBIC)

model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]

@torch.no_grad()
def embed(rows, bs=64):
    out = []
    for s in range(0, len(rows), bs):
        x = torch.stack([(torch.from_numpy(np.asarray(crop(r), np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
                         for r in rows[s:s + bs]])
        o = model(pixel_values=x).last_hidden_state
        out.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy())   # CLS + Mittel der Bildteile
        if (s // bs) % 20 == 0:
            print(f"  Merkmale {s + len(x)}/{len(rows)}", flush=True)
    return np.concatenate(out)

ALL_CACHE = os.path.join(RAW, "merkmale_dinov2_alle.npy")   # Zwischenspeicher bei den Rohdaten, nicht im Repo
if os.path.exists(ALL_CACHE):
    F_all = np.load(ALL_CACHE)
else:
    F_all = embed(list(df.itertuples())); np.save(ALL_CACHE, F_all)
MW = ["score", "size_rel", "elong_rel", "solidity_rel", "sharp_rel", "border", "parts"]
def feats(idx, kind):
    m = df.loc[idx, MW].values.astype(np.float32)
    hive = (df.loc[idx, "hive"] == "rot").values[:, None].astype(np.float32)
    if kind == "messwerte":
        return np.hstack([m, hive])
    if kind == "dinov2":
        return F_all[idx]
    return np.hstack([F_all[idx], m, hive])

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))

# ---------------------------------------------------------------- Ehrliche Prüfung (5-fach, 5 Wiederholungen)
y = L.gut.values; idx = L.index.values
oof = {}
for kind in ["messwerte", "dinov2", "beides"]:
    X = feats(idx, kind); P = np.zeros((5, len(y)))
    for rep in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            P[rep, te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof[kind] = P.mean(0)

# Regel-Filter (beste Einstellung je Stock aus optimierung.csv) zum Vergleich
REGEL = {"gelb": (0.607, 0.797), "rot": (0.727, 0.649)}
rows = []
for hive in ["gelb", "rot"]:
    s = (L.hive == hive).values
    for kind, p in oof.items():
        auc = roc_auc_score(y[s], p[s])
        # Genauigkeit bei derselben Ausbeute wie der Regel-Filter (fairer Vergleich)
        rec_target = REGEL[hive][1]
        thr = np.quantile(p[s][y[s] == 1], 1 - rec_target)
        kept = p[s] >= thr
        prec = y[s][kept].mean()
        rows.append(dict(stock=hive, merkmale=kind, auc=round(auc, 3), ausbeute=round(kept[y[s] == 1].mean(), 3),
                         genauigkeit=round(prec, 3), regel_genauigkeit=REGEL[hive][0], ohne_filter=round(y[s].mean(), 3)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "vergleich.csv"), index=False)
print("\nVergleich (gleiche Ausbeute wie Regel-Filter):\n" + R.to_string(index=False), flush=True)

# Lernkurve: bringen mehr Bewertungen noch etwas?
lc = []
X = feats(idx, "beides")
for frac in [0.25, 0.5, 0.75, 1.0]:
    aucs = []
    for rep in range(10):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            rng = np.random.default_rng(rep); tr = rng.choice(tr, max(int(len(tr) * frac), 20), replace=False)
            p = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    lc.append(dict(anteil_bewertungen=frac, n_training=int(len(y) * 0.8 * frac), auc=round(np.mean(aucs), 3)))
LC = pd.DataFrame(lc); LC.to_csv(os.path.join(OUT, "lernkurve.csv"), index=False)
print("\nLernkurve:\n" + LC.to_string(index=False), flush=True)

# Welche Fehlerart wird erkannt? (Anteil, der bei der obigen Schwelle rausfliegt)
L["p"] = oof["beides"]
gr = []
for hive in ["gelb", "rot"]:
    H = L[L.hive == hive]; thr = np.quantile(H.p[H.gut == 1], 1 - REGEL[hive][1])
    for g, G in H.groupby("grund"):
        gr.append(dict(stock=hive, grund=g, n=len(G), aussortiert_prozent=round(100 * (G.p < thr).mean())))
GR = pd.DataFrame(gr); GR.to_csv(os.path.join(OUT, "grund_erkannt.csv"), index=False)
print("\nWas der gelernte Filter aussortiert:\n" + GR.to_string(index=False), flush=True)

# ---------------------------------------------------------------- Bilder zum Ansehen
def sheet(sub, title, path, cols=10, n=40):
    sub = sub.head(n); rows_ = max(1, (len(sub) + cols - 1) // cols); t = 112
    im = Image.new("RGB", (cols * t, rows_ * (t + 14) + 24), "white"); d = ImageDraw.Draw(im); d.text((4, 4), title, fill="black")
    for k, r in enumerate(sub.itertuples()):
        x, y0 = (k % cols) * t, 24 + (k // cols) * (t + 14)
        im.paste(crop(r, size=t - 4), (x + 2, y0))
        d.text((x + 4, y0 + t - 2), f"{r.p:.2f} {r.grund[:12] if isinstance(r.grund, str) else ''}", fill="black")
    im.save(path)
for hive in ["gelb", "rot"]:
    H = L[L.hive == hive]; thr = np.quantile(H.p[H.gut == 1], 1 - REGEL[hive][1])
    sheet(H[(H.gut == 0) & (H.p >= thr)].sort_values("p", ascending=False),
          f"{hive}: SCHLECHT bewertet, aber vom Filter BEHALTEN (Zahl = Wahrscheinlichkeit gut)", os.path.join(OUT, f"{hive}_fehler_behalten.png"))
    sheet(H[(H.gut == 1) & (H.p < thr)].sort_values("p"),
          f"{hive}: GUT bewertet, aber vom Filter VERWORFEN", os.path.join(OUT, f"{hive}_fehler_verworfen.png"))

# ---------------------------------------------------------------- Auf alle Masken anwenden (Modell aus allen Bewertungen)
final = clf().fit(feats(idx, "beides"), y)
df["p_gut"] = final.predict_proba(feats(df.index.values, "beides"))[:, 1]
summ = []
for hive in ["gelb", "rot"]:
    H = df[df.hive == hive]; thr = np.quantile(L.p[(L.hive == hive) & (L.gut == 1)], 1 - REGEL[hive][1])
    summ.append(dict(stock=hive, masken=len(H), behalten=int((H.p_gut >= thr).sum()),
                     behalten_prozent=round(100 * (H.p_gut >= thr).mean(), 1), schwelle=round(thr, 3)))
    hs = H.copy(); hs["p"] = hs.p_gut
    sheet(hs.sort_values("p", ascending=False).iloc[::max(1, len(hs) // 40)].sort_values("p", ascending=False),
          f"{hive}: alle Masken, Stichprobe von bester zu schlechtester Wahrscheinlichkeit (Schwelle {thr:.2f})",
          os.path.join(OUT, f"{hive}_alle_sortiert.png"), n=40)
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "anwendung_alle.csv"), index=False)
df[["name", "hive", "p_gut"]].to_csv(os.path.join(OUT, "p_gut_alle.csv"), index=False)
print("\nAuf alle Masken angewendet:\n" + S.to_string(index=False))
print("FERTIG")
