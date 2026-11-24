# BeeReal Paper 1, Versuch 03. Code genau wie in versuche/03_blindtest_filter/notebooks/notebook_blindtest.ipynb ausgeführt.
# BeeReal Blindtest: neue SAM-3-Masken aus Train-Sequenzen, die in Jans 300 Bewertungen NICHT vorkommen.
# Nur Rohdaten (Bretter + Masken); Messwerte, Filter und Bewertungsordner werden lokal berechnet.
SEQS = ["rec2123_seq030", "rec2123_seq052", "rec2125_seq003", "rec2125_seq007"]
EVERY = 120         # alle 2 s ein Bild -> 15 Bilder pro Sequenz
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
log("FERTIG")
