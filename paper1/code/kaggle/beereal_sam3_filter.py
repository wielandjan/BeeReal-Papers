# BeeReal Paper 1, Versuch 02. Code genau wie in versuche/02_masken_bewertung_und_filter/notebooks/notebook_sam3_filter_v2.ipynb ausgeführt.
# BeeReal SAM 3 + Filter: Bienen mit SAM 3 ausschneiden (Maske), gerade drehen, filtern.
# Ergebnis: Trichter-Tabelle (wie viele fallen durch welchen Filter), alle Bilder als ZIP, Stichproben-Bögen.
# Nur Train-Sequenzen, Val und Test bleiben unberührt.
SEQS = ["rec2123_seq015", "rec2123_seq041", "rec2125_seq001", "rec2125_seq005"]
EVERY = 60          # 1 Bild pro Sekunde (Video hat 59,94 fps) -> 30 Bilder pro Sequenz
# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "torchcodec"], check=False)
# %%
import glob, re
from huggingface_hub import login
_f = glob.glob("/kaggle/input/**/hf_token*", recursive=True)
if not _f:
    _f = [p for p in glob.glob("/kaggle/input/**/*", recursive=True) if p.endswith((".txt", ".csv"))]
login(re.search(r"hf_[A-Za-z0-9]+", open(_f[0], encoding="utf-8-sig").read()).group(0))
print("HF_LOGIN OK", flush=True)
# %%
import io, json, os, random, shutil, time, traceback, zipfile
import cv2, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder
from transformers import Sam3Processor, Sam3Model

START = time.time()
OUT = "/kaggle/working/"
IMG = "/kaggle/temp/bilder/"
os.makedirs(IMG, exist_ok=True)
HIVE = {"2123": "gelb", "2125": "rot"}
ROI = {"2123": (90, 800, 1080, 1025), "2125": (0, 750, 1080, 1035)}
START_FRAME = {"rec2125_seq000": 900}

# Vorläufige Schwellen. Sie werden nach Sichtprüfung der Bögen angepasst.
TH = dict(score=0.5, min_area=40, border_px=2, size_lo=0.4, size_hi=1.8, overlap=0.05,
          elong=1.5, solidity=0.80, sharp_pct=20)

def log(*a):
    print(f"[{(time.time()-START)/60:6.1f} min]", *a, flush=True)

proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to("cuda:0").eval()
log("SAM 3 geladen")
# %%
# Schritt 1: SAM 3 auf jedem 60. Bild. Speichert pro Biene Maske (nur im Kasten-Bereich), Kasten, Sicherheit.
ROWS, MASKS, BOARDS, TIMES = [], {}, {}, []

def run_sam3(board):
    inp = proc(images=board, text="bee", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model(**inp)
    r = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                target_sizes=inp.get("original_sizes").tolist())[0]
    return r["masks"].cpu().numpy() > 0.5, r["boxes"].cpu().numpy(), r["scores"].cpu().float().numpy()

for sid in SEQS:
    rec = sid[3:7]
    path = hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset")
    dec = VideoDecoder(path)
    n = len(dec)
    for f in range(START_FRAME.get(sid, 0), n, EVERY):
        board = Image.fromarray(dec.get_frame_at(f).data.permute(1, 2, 0).numpy()).crop(ROI[rec])
        t0 = time.time(); masks, boxes, scores = run_sam3(board); TIMES.append(time.time() - t0)
        key = f"{sid}_f{f:04d}"; BOARDS[key] = np.array(board)
        # Wie viele Masken überdecken jeden Pixel? Für den Überlappungsfilter.
        count = masks.sum(0) if len(masks) else np.zeros(board.size[::-1], int)
        for i, (m, b, s) in enumerate(zip(masks, boxes, scores)):
            area = int(m.sum())
            ov = float((m & (count > 1)).sum() / max(area, 1))
            ROWS.append(dict(key=key, sid=sid, hive=HIVE[rec], frame=f, i=i, score=float(s),
                             x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), area=area, overlap=ov))
            MASKS[(key, i)] = m
    log(sid, "Bilder:", len(range(START_FRAME.get(sid, 0), n, EVERY)), "Bienen bisher:", len(ROWS))
del model; torch.cuda.empty_cache()
df = pd.DataFrame(ROWS)
log("SAM 3 fertig. Sekunden pro Bild (Median):", round(float(np.median(TIMES)), 2))

# Rohdaten sichern (Bretter + Masken), damit Filter-Varianten lokal ohne GPU durchgerechnet werden können.
RAW = "/kaggle/temp/rohdaten/"; os.makedirs(RAW + "bretter", exist_ok=True)
for key, b in BOARDS.items():
    Image.fromarray(b).save(RAW + f"bretter/{key}.png")
packed = {}
for (key, i), m in MASKS.items():
    ys, xs = np.nonzero(m)
    if not len(ys):
        continue
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    packed[f"{key}|{i}|{y0}|{x0}|{y1 - y0}|{x1 - x0}"] = np.packbits(m[y0:y1, x0:x1])
np.savez_compressed(RAW + "masken.npz", **packed)
df.to_csv(RAW + "sam3_roh.csv", index=False)
shutil.make_archive(OUT + "rohdaten", "zip", RAW)
log("Rohdaten:", round(os.path.getsize(OUT + "rohdaten.zip") / 1e6, 1), "MB")
# %%
# Schritt 2: Messwerte pro Maske: Teile, Rand, Form (Längsachse, Kompaktheit), Ausrichtung, Schärfe.
def shape_stats(m):
    ys, xs = np.nonzero(m)
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    parts = int((st[1:, cv2.CC_STAT_AREA] >= 0.1 * m.sum()).sum()) if n_lab > 1 else 0
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 0
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    ev, evec = np.linalg.eigh(np.cov((pts - c).T)) if len(pts) > 2 else (np.array([1, 1]), np.eye(2))
    elong = float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    v = evec[:, 1]                                       # Längsachse
    return dict(parts=parts, solidity=float(m.sum() / max(hull, 1)), elong=elong,
                cx=float(c[0]), cy=float(c[1]), angle=float(np.degrees(np.arctan2(v[1], v[0]))),
                length=float(4 * np.sqrt(ev[1])))

def aligned_crop(board, m, st, out=96):
    """Dreht die Biene so, dass die Längsachse senkrecht steht und der größere Teil (Hinterleib) unten ist.
    Hintergrund wird schwarz. Kopf-oben ist eine Faustregel und muss geprüft werden."""
    side = int(max(st["length"] * 1.3, 16))
    best = None
    for ang in (st["angle"] - 90, st["angle"] + 90):
        M = cv2.getRotationMatrix2D((st["cx"], st["cy"]), ang, 1.0)
        M[:, 2] += [side / 2 - st["cx"], side / 2 - st["cy"]]
        mm = cv2.warpAffine(m.astype(np.uint8), M, (side, side), flags=cv2.INTER_NEAREST)
        xs = np.nonzero(mm)[1]
        w = (xs.max() - xs.min()) if len(xs) else 1e9
        if best is None or w < best[0]:
            best = (w, M, mm)
    _, M, mm = best
    im = cv2.warpAffine(board, M, (side, side), flags=cv2.INTER_LINEAR)
    if mm[: side // 2].sum() > mm[side // 2:].sum():   # größerer Teil oben -> um 180° drehen
        im, mm = im[::-1, ::-1], mm[::-1, ::-1]
    im = im * mm[..., None]
    sc = out / side
    im = cv2.resize(np.ascontiguousarray(im), (out, out), interpolation=cv2.INTER_AREA)
    mm = cv2.resize(np.ascontiguousarray(mm), (out, out), interpolation=cv2.INTER_NEAREST)
    return im, mm

def sharpness(im, mm):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    inner = cv2.erode(mm, np.ones((5, 5), np.uint8))     # Rand der Maske nicht mitzählen
    return float(lap[inner > 0].var()) if inner.sum() > 20 else 0.0

stats, ALIGNED = [], {}
for r in df.itertuples():
    m = MASKS[(r.key, r.i)]; b = BOARDS[r.key]
    if r.area < TH["min_area"]:
        stats.append(dict(parts=0, solidity=0, elong=0, border=False, sharp=0)); continue
    st = shape_stats(m)
    H, W = m.shape; p = TH["border_px"]
    border = bool(m[:p].any() or m[-p:].any() or m[:, :p].any() or m[:, -p:].any())
    im, mm = aligned_crop(b, m, st)
    ALIGNED[(r.key, r.i)] = im
    stats.append(dict(parts=st["parts"], solidity=st["solidity"], elong=st["elong"], border=border,
                      sharp=sharpness(im, mm)))
df = pd.concat([df, pd.DataFrame(stats)], axis=1)
df["size_rel"] = df.area / df.groupby("hive").area.transform("median")
log("Messwerte fertig")
# %%
# Schritt 3: Filter nacheinander (Trichter). Jede Biene bekommt den ersten Filter, an dem sie scheitert.
sharp_th = df[df.area >= TH["min_area"]].groupby("hive").sharp.quantile(TH["sharp_pct"] / 100).to_dict()
FILTERS = [
    ("1_unsicher",      f"SAM-3-Sicherheit < {TH['score']}",                          lambda d: d.score < TH["score"]),
    ("2_maske_kaputt",  f"Maske < {TH['min_area']} px oder in mehrere Teile zerfallen", lambda d: (d.area < TH["min_area"]) | (d.parts > 1)),
    ("3_am_rand",       "Maske berührt den Rand des Brett-Ausschnitts (abgeschnitten)", lambda d: d.border),
    ("4_groesse",       f"Fläche < {TH['size_lo']}× oder > {TH['size_hi']}× typische Biene (Teilstück oder mehrere Bienen)",
                        lambda d: (d.size_rel < TH["size_lo"]) | (d.size_rel > TH["size_hi"])),
    ("5_ueberlappung",  f"mehr als {int(TH['overlap']*100)} % der Maske von anderer Maske überdeckt", lambda d: d.overlap > TH["overlap"]),
    ("6_form",          f"nicht länglich (Längen-Verhältnis < {TH['elong']}) oder ausgefranst (Kompaktheit < {TH['solidity']}): "
                        "von vorn/von der Seite/verdeckt", lambda d: (d.elong < TH["elong"]) | (d.solidity < TH["solidity"])),
    ("7_unscharf",      f"Schärfe unter dem {TH['sharp_pct']}. Perzentil des Stocks", lambda d: d.sharp < d.hive.map(sharp_th)),
]
df["ergebnis"] = "behalten"
for name, _, fn in FILTERS[::-1]:
    df.loc[fn(df), "ergebnis"] = name
for name, _, fn in FILTERS:
    df["allein_" + name] = fn(df)

rows = []
for hive in ["gelb", "rot", "alle"]:
    d = df if hive == "alle" else df[df.hive == hive]
    left = len(d); rows.append(dict(stock=hive, schritt="0_sam3_gefunden", beschreibung="alle Masken von SAM 3 (Sicherheit ≥ 0,3)",
                                    weg=0, weg_prozent=0.0, uebrig=left, uebrig_prozent=100.0, allein_weg_prozent=0.0))
    for name, desc, _ in FILTERS:
        k = int((d.ergebnis == name).sum()); left -= k
        rows.append(dict(stock=hive, schritt=name, beschreibung=desc, weg=k, weg_prozent=round(100 * k / len(d), 1),
                         uebrig=left, uebrig_prozent=round(100 * left / len(d), 1),
                         allein_weg_prozent=round(100 * d["allein_" + name].mean(), 1)))
FUNNEL = pd.DataFrame(rows)
FUNNEL.to_csv(OUT + "trichter.csv", index=False)
df.drop(columns=[c for c in df.columns if c.startswith("allein_")]).to_csv(OUT + "bienen.csv", index=False)
print(FUNNEL.to_string(index=False), flush=True)
# %%
# Schritt 4: Bilder. Behaltene: gedreht, Hintergrund schwarz. Aussortierte: Ausschnitt mit Umriss (damit man sieht, warum).
def raw_with_outline(r, pad=0.4):
    b = BOARDS[r.key]; m = MASKS[(r.key, r.i)]
    w, h = r.x2 - r.x1, r.y2 - r.y1
    x1, y1 = int(max(r.x1 - pad * w, 0)), int(max(r.y1 - pad * h, 0))
    x2, y2 = int(min(r.x2 + pad * w, b.shape[1])), int(min(r.y2 + pad * h, b.shape[0]))
    im = b[y1:y2, x1:x2].copy()
    cnts, _ = cv2.findContours(m[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnts, -1, (255, 0, 255), 1)
    return im

def picture(r):
    return ALIGNED[(r.key, r.i)] if r.ergebnis == "behalten" else raw_with_outline(r)

for r in df.itertuples():
    if r.x2 - r.x1 < 2 or r.y2 - r.y1 < 2:
        continue
    d = os.path.join(IMG, r.hive, r.ergebnis); os.makedirs(d, exist_ok=True)
    Image.fromarray(picture(r)).save(os.path.join(d, f"{r.key}_b{r.i:03d}.png"))
shutil.make_archive(OUT + "alle_bilder", "zip", IMG)
log("ZIP:", round(os.path.getsize(OUT + "alle_bilder.zip") / 1e6, 1), "MB")

def sheet(d, title, fname, n=60, cols=12):
    d = d.sample(min(n, len(d)), random_state=0) if len(d) else d
    if d.empty:
        return
    rows_ = int(np.ceil(len(d) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(cols * 1.3, rows_ * 1.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, d.itertuples()):
        ax.imshow(picture(r)); ax.set_title(f"s{r.score:.2f} e{r.elong:.1f}\nk{r.solidity:.2f} sch{r.sharp:.0f}", fontsize=6)
    fig.suptitle(title, fontsize=10); plt.tight_layout()
    plt.savefig(OUT + fname, dpi=90); plt.close()

for hive in ["gelb", "rot"]:
    for name in ["behalten"] + [f[0] for f in FILTERS]:
        d = df[(df.hive == hive) & (df.ergebnis == name)]
        sheet(d, f"{hive}: {name} ({len(d)} Bienen, Stichprobe)", f"bogen_{hive}_{name}.png")

# Übersicht: ein ganzes Brett, Umriss grün = behalten, rot = aussortiert, mit Filternummer
COL = {"behalten": (0, 255, 0)}
for key in [k for k in BOARDS if k.endswith("_f0600")]:
    im = BOARDS[key].copy()
    for r in df[df.key == key].itertuples():
        cnts, _ = cv2.findContours(MASKS[(key, r.i)].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = COL.get(r.ergebnis, (255, 0, 0))
        cv2.drawContours(im, cnts, -1, c, 2)
        if r.ergebnis != "behalten":
            cv2.putText(im, r.ergebnis[0], (int(r.x1), int(r.y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    Image.fromarray(im).save(OUT + f"uebersicht_{key}.png")
log("Bilder fertig")
# %%
md = ["# SAM 3 + Filter", "", f"Sequenzen: {', '.join(SEQS)} (nur Train), je 1 Bild pro Sekunde. "
      f"SAM 3: Median {np.median(TIMES):.2f} s pro Bild.", "",
      "Trichter: Jede Biene zählt beim ersten Filter, an dem sie scheitert. "
      "`allein_weg_prozent` = so viele würde der Filter allein entfernen.", "",
      FUNNEL.to_string(index=False), "", "Schwellen (vorläufig): " + json.dumps(TH), "",
      "Schärfe-Schwelle je Stock: " + json.dumps({k: round(v, 1) for k, v in sharp_th.items()})]
open(OUT + "REPORT.md", "w", encoding="utf-8").write("\n".join(md))
json.dump(dict(sam3_bienen_pro_bild={k: int((df.key == k).sum()) for k in BOARDS}, sekunden_pro_bild=TIMES), open(OUT + "sam3.json", "w"))
pd.DataFrame([dict(key=r.key, i=r.i, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, score=r.score) for r in df.itertuples()]).to_csv(OUT + "sam3_kaesten.csv", index=False)
print("\n".join(md), flush=True)
log("FERTIG")
