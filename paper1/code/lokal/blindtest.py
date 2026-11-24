# BeeReal Paper 1, Versuch 03.
"""Blindtest der Qualitätsfilter auf neuen Masken (lokal, CPU reicht).

Neue Zufallsmasken aus Sequenzen, die in Jans erster Bewertung nicht vorkamen. Jan bewertet sie, ohne die
Filterentscheidung zu kennen. Danach wird gezählt, wie viel Prozent der behaltenen Masken gut sind:
ohne Filter, beste feste Regel (aus der ersten Bewertung) und gelernter Filter (gelernt nur aus der ersten Bewertung).

Vorher:  python filter_auswertung.py <neu-rohdaten> <neu-ausgabe> --n_bewerten=100 --regel=<alt-ausgabe>/optimierung.csv

Aufruf:  python blindtest.py vorbereiten <alt-rohdaten> <alt-ausgabe> <neu-rohdaten> <neu-ausgabe>
         python blindtest.py auswerten <neu-ausgabe>
"""
import os, shutil, sys
import numpy as np
import pandas as pd

GRUENDE = ["1_mehrere_bienen", "2_nur_teil_der_biene", "3_umriss_falsch", "4_am_rand_abgeschnitten", "5_verdeckt",
           "6_von_vorn_oder_seite", "7_unscharf", "8_keine_biene", "9_sonstiges"]
REGEL_AUSBEUTE = {"gelb": 0.797, "rot": 0.649}   # Ausbeute der besten Regel in der ersten Bewertung (fairer Vergleich)
MW = ["score", "size_rel", "elong_rel", "solidity_rel", "sharp_rel", "border", "parts"]


def labels(bew):
    """Liest die Bewertung aus den Ordnern <stock>/sehr_gut, <stock>/gut, <stock>/schlecht/<grund>."""
    lab, grund = {}, {}
    for root, _, files in os.walk(bew):
        parts = os.path.relpath(root, bew).replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        for f in files:
            if not f.endswith(".png") or " - Kopie" in f:
                continue
            n = f[:-4]
            if parts[1] == "schlecht":
                lab[n] = 0; grund[n] = parts[2] if len(parts) > 2 else "ohne_grund"
            elif parts[1] in ("gut", "sehr_gut"):
                lab[n] = 1; grund[n] = parts[1]
    return lab, grund


def name_col(df):
    return [f"{h}_{k}_b{i:03d}" for h, k, i in zip(df.hive, df.key, df.i)]


if sys.argv[1] == "vorbereiten":
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel

    RAW_A, VAR_A, RAW_N, VAR_N = sys.argv[2:6]
    torch.set_num_threads(os.cpu_count())
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def embed(df, raw, bs=64):
        boards = {}
        def crop(r, pad=0.25, size=224):   # identisch zu qualitaet_klassifikator.py
            if r.key not in boards:
                boards[r.key] = Image.open(os.path.join(raw, "bretter", r.key + ".png")).convert("RGB")
            w, h = r.x2 - r.x1, r.y2 - r.y1; s = max(w, h) * 1.5
            cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
            return boards[r.key].crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((size, size), Image.BICUBIC)
        rows, out = list(df.itertuples()), []
        with torch.no_grad():
            for s in range(0, len(rows), bs):
                x = torch.stack([(torch.from_numpy(np.asarray(crop(r), np.float32) / 255).permute(2, 0, 1) - MEAN) / STD
                                 for r in rows[s:s + bs]])
                o = model(pixel_values=x).last_hidden_state
                out.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy())
                if (s // bs) % 20 == 0:
                    print(f"  Merkmale {s + len(x)}/{len(rows)}", flush=True)
        return np.concatenate(out)

    def feats(df, F):
        return np.hstack([F, df[MW].values.astype(np.float32), (df.hive == "rot").values[:, None].astype(np.float32)])

    def clf():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))

    # Alte Bewertung = Lehrer. Merkmale aus dem Zwischenspeicher von qualitaet_klassifikator.py.
    A = pd.read_csv(os.path.join(VAR_A, "bienen_messwerte.csv")); A["name"] = name_col(A)
    FA = np.load(os.path.join(RAW_A, "merkmale_dinov2_alle.npy"))
    lab, _ = labels(os.path.join(VAR_A, "bewerten"))
    A["gut"] = A.name.map(lab); L = A[A.gut.notna()]
    X, y = feats(L, FA[L.index.values]), L.gut.values.astype(int)
    print("Alte Bewertung:", len(L))

    # Schwelle wie in qualitaet_klassifikator.py: gleiche Ausbeute wie die Regel, aus ehrlichen Vorhersagen
    P = np.zeros((5, len(y)))
    for rep in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
            P[rep, te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof = P.mean(0)
    THR = {h: float(np.quantile(oof[(L.hive == h).values & (y == 1)], 1 - REGEL_AUSBEUTE[h])) for h in ["gelb", "rot"]}
    print("Schwellen:", {k: round(v, 3) for k, v in THR.items()})
    final = clf().fit(X, y)

    # Neue Masken
    N = pd.read_csv(os.path.join(VAR_N, "bienen_messwerte.csv")); N["name"] = name_col(N)
    cache = os.path.join(RAW_N, "merkmale_dinov2_alle.npy")
    FN = np.load(cache) if os.path.exists(cache) else embed(N, RAW_N)
    np.save(cache, FN)
    N["p_gut"] = final.predict_proba(feats(N, FN))[:, 1]
    N["modell_behalten"] = N.p_gut >= N.hive.map(THR)
    N["regel_behalten"] = N.regel_best == "behalten"
    # Liegt bewusst NICHT im Bewertungsordner, damit die Bewertung blind bleibt
    N[["name", "hive", "key", "i", "p_gut", "modell_behalten", "regel_behalten", "regel_best"]].to_csv(
        os.path.join(VAR_N, "filter_entscheidung_NICHT_VOR_DEM_BEWERTEN_OEFFNEN.csv"), index=False)
    for h, g in N.groupby("hive"):
        print(h, "Masken:", len(g), "Modell behält:", round(g.modell_behalten.mean(), 3), "Regel behält:", round(g.regel_behalten.mean(), 3))

    # Bewertungsordner: Bilder nach Stock, Zielordner schon angelegt
    bew = os.path.join(VAR_N, "bewerten")
    for f in [f for f in os.listdir(bew) if f.endswith(".png")]:
        hive = f.split("_")[0]
        os.makedirs(os.path.join(bew, hive), exist_ok=True)
        shutil.move(os.path.join(bew, f), os.path.join(bew, hive, f))
    for hive in ["gelb", "rot"]:
        for d in ["sehr_gut", "gut"] + [os.path.join("schlecht", g) for g in GRUENDE]:
            os.makedirs(os.path.join(bew, hive, d), exist_ok=True)
    shutil.rmtree(os.path.join(bew, "schlecht"), ignore_errors=True)   # leerer Ordner aus filter_auswertung.py
    print("FERTIG")

elif sys.argv[1] == "auswerten":
    VAR_N = sys.argv[2]
    N = pd.read_csv(os.path.join(VAR_N, "filter_entscheidung_NICHT_VOR_DEM_BEWERTEN_OEFFNEN.csv"))
    lab, grund = labels(os.path.join(VAR_N, "bewerten"))
    N["gut"] = N.name.map(lab); N["grund"] = N.name.map(grund)
    L = N[N.gut.notna()].copy()
    print("Bewertet:", len(L), "nicht bewertet:", int(N.name.isin(
        [f[:-4] for h in ["gelb", "rot"] for f in os.listdir(os.path.join(VAR_N, "bewerten", h)) if f.endswith(".png")]).sum()))
    rng = np.random.default_rng(0); rows = []
    for hive, H in L.groupby("hive"):
        for filt, keep in [("ohne Filter", np.ones(len(H), bool)), ("feste Regel", H.regel_behalten.values),
                           ("gelernter Filter", H.modell_behalten.values)]:
            g = H.gut.values
            boot = [g[i][keep[i]].mean() for i in (rng.integers(0, len(H), len(H)) for _ in range(2000)) if keep[i].any()]
            rows.append(dict(stock=hive, filter=filt, bewertet=len(H), behalten=int(keep.sum()),
                             gut_unter_behaltenen=round(g[keep].mean(), 3),
                             spanne_95=f"{np.quantile(boot, 0.025):.2f}-{np.quantile(boot, 0.975):.2f}",
                             gute_behalten=round(keep[g == 1].mean(), 3)))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(VAR_N, "blindtest_ergebnis.csv"), index=False)
    print(R.to_string(index=False))
    # Beide Filter gegeneinander auf denselben Bildern: wo sind sie sich uneinig?
    ct = pd.crosstab([L.hive, L.regel_behalten, L.modell_behalten], L.gut)
    ct.to_csv(os.path.join(VAR_N, "blindtest_uneinig.csv")); print(ct)
    G = pd.crosstab([L.hive, L.grund], [L.regel_behalten.map({True: "Regel behält", False: "Regel weg"}),
                                       L.modell_behalten.map({True: "Modell behält", False: "Modell weg"})])
    G.to_csv(os.path.join(VAR_N, "blindtest_grund.csv")); print(G)
