# BeeReal Paper 1, Versuch 10 (SAM-3-Vorschläge für die Val-Ausschnitte).
# BeeReal: SAM-3-Kästen für die Val-Ground-Truth (CVAT-Vorschläge).
# Je ein 15-s-Ausschnitt pro Fold-Block (Fold-Vorschlag 13.11.): Gelb G1–G4, Rot R1–R4. 5 Bilder/s = jedes 12. Bild.
# Gespeichert: pro Bild Sicherheit + Kasten im ROI-Ausschnitt (keine Masken). Tracker und CVAT-Pakete laufen lokal.
CLIPS = {"rec2123_seq021": "G1", "rec2123_seq036": "G2", "rec2123_seq051": "G3", "rec2123_seq073": "G4",
         "rec2125_seq001": "R1", "rec2125_seq005": "R2", "rec2125_seq009": "R3", "rec2125_seq012": "R4"}
START_S, DAUER_S, STEP = 0, 15, 12
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
import os, sys, json, time
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder
from transformers import Sam3Processor, Sam3Model
OUT, START_S, DAUER_S, STEP = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); sids = sys.argv[5:]
ROI = {"2123": (90, 800, 1080, 1025), "2125": (0, 750, 1080, 1035)}
FPS = 59.94
proc = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to("cuda:0").eval()
for sid in sids:
    t0 = time.time()
    dec = VideoDecoder(hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset"))
    frames = list(range(int(START_S * FPS), int((START_S + DAUER_S) * FPS), STEP))
    out = {}
    for f in frames:
        board = Image.fromarray(dec.get_frame_at(f).data.permute(1, 2, 0).numpy()).crop(ROI[sid[3:7]])
        inp = proc(images=board, text="bee", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            o = model(**inp)
        r = proc.post_process_instance_segmentation(o, threshold=0.3, mask_threshold=0.5, target_sizes=inp.get("original_sizes").tolist())[0]
        out[f] = [[round(float(s), 3)] + [round(float(x), 1) for x in b] for b, s in zip(r["boxes"].cpu().tolist(), r["scores"].cpu().float().tolist())]
    name = f"bild_{sid}_{int(START_S)}-{int(START_S + DAUER_S)}s.json"
    json.dump(out, open(os.path.join(OUT, name), "w"))
    print(sid, len(frames), "Bilder", round(time.time() - t0), "s", round(sum(len(v) for v in out.values()) / len(out), 1), "Kästen/Bild", flush=True)
"""
# %%
import os, subprocess, sys, threading, time
START = time.time(); OUT = "/kaggle/working/"
open(OUT + "worker.py", "w").write(WORKER)
sids = list(CLIPS)
teile = [sids[0::2], sids[1::2]]   # je Grafikkarte die Hälfte, Gelb und Rot gemischt
def lauf(gpu, s):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    r = subprocess.run([sys.executable, OUT + "worker.py", OUT, str(START_S), str(DAUER_S), str(STEP)] + s, env=env, capture_output=True, text=True)
    print(f"GPU {gpu}:", r.stdout, (r.stderr or "")[-800:] if r.returncode else "", flush=True)
ths = [threading.Thread(target=lauf, args=(g, t)) for g, t in enumerate(teile)]
[t.start() for t in ths]; [t.join() for t in ths]
print("FERTIG", round((time.time() - START) / 60, 1), "min", sorted(f for f in os.listdir(OUT) if f.startswith("bild_")), flush=True)
