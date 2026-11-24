# BeeReal Paper 1, Versuch 08. Code genau wie in versuche/08_sam3_alle_sequenzen/notebooks/notebook_stufe_b.ipynb ausgeführt.
# BeeReal Stufe B: SAM 3 (Einzelbild, Text "bee") auf allen Train/Val-Sequenzen des 3-Fold-Vorschlags (13.11.).
# Test (Gelb 060–064, 082–089; Rot 015–020) und Lücken werden absichtlich NICHT verarbeitet.
# 10 Bilder/s (jedes 6. Bild). Pro Biene: Sicherheit, Kasten (ROI-Koordinaten), Maskenfläche, Füllgrad, Überlappung.
# Gespeichert wird nach jeder Sequenz (eine JSON pro Sequenz). Tracker, Ausschnitte und Filter laufen danach separat.
GELB = list(range(15, 34)) + list(range(35, 54)) + list(range(55, 59)) + list(range(66, 81))   # G1, G2, G3
ROT = list(range(0, 4)) + list(range(5, 9)) + list(range(10, 14))                              # R1, R2, R3
SEQS = [f"rec2123_seq{i:03d}" for i in GELB] + [f"rec2125_seq{i:03d}" for i in ROT]
STEP = 6
START_FRAME = {"rec2125_seq000": 900}
STOP_NACH_H = 11.0     # danach keine neue Sequenz mehr anfangen (Kaggle-Grenze 12 h)
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
WORKER = r"""
import os, sys, json, time, traceback
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder
from transformers import Sam3Processor, Sam3Model
OUT, STEP, ENDE = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); START_FRAME = json.loads(sys.argv[4]); sids = sys.argv[5:]
ROI = {"2123": (90, 800, 1080, 1025), "2125": (0, 750, 1080, 1035)}
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to("cuda:0").eval()
for sid in sids:
    ziel = os.path.join(OUT, f"sam3_{sid}.json")
    if os.path.exists(ziel):
        continue
    if time.time() > ENDE:
        print("ZEIT", sid, "nicht mehr begonnen", flush=True); continue
    try:
        t0 = time.time()
        path = hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset")
        dec = VideoDecoder(path); n = len(dec); out = {}
        for f in range(START_FRAME.get(sid, 0), n, STEP):
            board = Image.fromarray(dec.get_frame_at(f).data.permute(1, 2, 0).numpy()).crop(ROI[sid[3:7]])
            inp = proc(images=board, text="bee", return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                o = model(**inp)
            r = proc.post_process_instance_segmentation(o, threshold=0.3, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist())[0]
            m = (r["masks"] > 0.5)
            zeilen = []
            if len(m):
                count = m.sum(0)
                flaeche = m.flatten(1).sum(1)
                ueber = (m & (count > 1)).flatten(1).sum(1)
                for b, s, a, u in zip(r["boxes"].tolist(), r["scores"].float().tolist(), flaeche.tolist(), ueber.tolist()):
                    box_a = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
                    zeilen.append([round(s, 3)] + [round(x, 1) for x in b] + [int(a), round(a / box_a, 3), round(u / max(a, 1), 3)])
            out[f] = zeilen
        json.dump(dict(sid=sid, step=STEP, roi=ROI[sid[3:7]], felder=["score", "x1", "y1", "x2", "y2", "flaeche", "fuellgrad", "ueberlappung"], bilder=out), open(ziel + ".tmp", "w"))
        os.replace(ziel + ".tmp", ziel)
        print("OK", sid, len(out), "Bilder", round(time.time() - t0), "s", round(sum(len(v) for v in out.values()) / max(len(out), 1), 1), "Bienen/Bild", flush=True)
    except Exception:
        print("FEHLER", sid, traceback.format_exc()[-600:], flush=True)
"""
# %%
import json, os, subprocess, sys, threading, time
START = time.time(); OUT = "/kaggle/working/stufe_b/"; os.makedirs(OUT, exist_ok=True)
open("/kaggle/working/worker.py", "w").write(WORKER)
ENDE = START + STOP_NACH_H * 3600
# Abwechselnd verteilen, damit beide Karten gleich viele Gelb- und Rot-Sequenzen bekommen
teile = [SEQS[0::2], SEQS[1::2]]
def lauf(gpu, s):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.Popen([sys.executable, "/kaggle/working/worker.py", OUT, str(STEP), str(ENDE), json.dumps(START_FRAME)] + s,
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for zeile in p.stdout:
        if zeile.startswith(("OK", "FEHLER", "ZEIT", "Traceback")):
            print(f"[{(time.time() - START) / 60:6.1f} min] GPU {gpu}:", zeile.rstrip(), flush=True)
    p.wait()
ths = [threading.Thread(target=lauf, args=(g, t)) for g, t in enumerate(teile)]
[t.start() for t in ths]; [t.join() for t in ths]
fertig = sorted(f for f in os.listdir(OUT) if f.endswith(".json"))
print("FERTIG", round((time.time() - START) / 3600, 2), "h,", len(fertig), "von", len(SEQS), "Sequenzen", flush=True)
json.dump(dict(fertig=fertig, fehlend=[s for s in SEQS if f"sam3_{s}.json" not in fertig]), open("/kaggle/working/stufe_b_info.json", "w"), indent=1)
