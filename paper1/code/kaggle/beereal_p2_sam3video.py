# BeeReal Paper 1, Versuch 06 (SAM 3 Einzelbild und Videomodus).
# BeeReal Versuch P2/P4: SAM 3 im Videomodus (eigener Tracker mit Gedächtnis) gegen SAM 3 im Einzelbildmodus.
# Gleiche Clips wie der Bildraten-Test: erste 10 s von Gelb seq065 und Rot seq012, 10 Bilder/s (jedes 6. Bild).
# Version 3: Jeder Versuch läuft in einem eigenen Prozess auf einer eigenen Grafikkarte (Version 2 war durch Fehler im selben Prozess verfälscht).
# Version 2: Der erste Lauf scheiterte im Videomodus an vollem Grafikspeicher (T4, fp32). Jetzt fp16 und eine Leiter aus 100, 50 und 25 Bildern.
# Der Einzelbildmodus ist schon ausgewertet (Version 1) und wird übersprungen.
# Gespeichert werden nur Kästen, Sicherheit und Objekt-ID je Bild (keine Masken), ausgewertet wird lokal.
CLIPS = {"rec2123_seq065": (90, 800, 1080, 1025), "rec2125_seq012": (0, 750, 1080, 1035)}
STEP = 6
KURZ = list(range(0, 600, STEP))       # 10 s
LANG = list(range(0, 1800, STEP))      # 30 s
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
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers', 'torchcodec'], check=False)
# %%
import glob, re
from huggingface_hub import login
_f = glob.glob('/kaggle/input/**/hf_token*', recursive=True)
if not _f:
    _f = [p for p in glob.glob('/kaggle/input/**/*', recursive=True) if p.endswith(('.txt', '.csv'))]
login(re.search(r'hf_[A-Za-z0-9]+', open(_f[0], encoding='utf-8-sig').read()).group(0))
print('HF_LOGIN OK', flush=True)
# %%
WORKER = r"""
import os, sys, json, time, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder
from transformers import Sam3VideoModel, Sam3VideoProcessor
sid, n, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ROI = {"rec2123_seq065": (90, 800, 1080, 1025), "rec2125_seq012": (0, 750, 1080, 1035)}[sid]
frames = list(range(0, 600, 6))[:n]
dec = VideoDecoder(hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset"))
ims = [Image.fromarray(dec.get_frame_at(f).data.permute(1, 2, 0).numpy()).crop(ROI) for f in frames]
dev = "cuda:0"
proc = Sam3VideoProcessor.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", dtype=torch.float16).to(dev).eval()
torch.cuda.reset_peak_memory_stats(dev); t0 = time.time()
sess = proc.init_video_session(video=ims, inference_device=dev, processing_device="cpu", video_storage_device="cpu", dtype=torch.float16)
sess = proc.add_text_prompt(inference_session=sess, text="bee")
out = {}
with torch.no_grad():
    for mo in model.propagate_in_video_iterator(inference_session=sess):
        p = proc.postprocess_outputs(sess, mo)
        out[frames[mo.frame_idx]] = [[int(i), round(float(s), 3)] + [round(float(x), 1) for x in b] for i, s, b in zip(p["object_ids"].tolist(), p["scores"].tolist(), p["boxes"].tolist())]
        print("Bild", mo.frame_idx, "Objekte", len(p["object_ids"]), flush=True)
json.dump(out, open(os.path.join(OUT, f"video_{sid}_{n}bilder.json"), "w"))
json.dump(dict(sid=sid, bilder=n, sekunden=round(time.time() - t0, 1), gpu_gb=round(torch.cuda.max_memory_allocated(dev) / 1e9, 2),
               objekte_gesamt=len({o[0] for r in out.values() for o in r})), open(os.path.join(OUT, f"ok_{sid}_{n}.json"), "w"))
"""
# %%
import json, os, subprocess, sys, threading, time
START = time.time(); OUT = "/kaggle/working/"; INFO = []
open(OUT + "worker.py", "w").write(WORKER)
def log(*a):
    print(f"[{(time.time()-START)/60:6.1f} min]", *a, flush=True)

def leiter(sid, gpu, ns):
    for n in ns:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        r = subprocess.run([sys.executable, OUT + "worker.py", sid, str(n), OUT], env=env, capture_output=True, text=True)
        ok = os.path.exists(OUT + f"ok_{sid}_{n}.json")
        INFO.append(dict(sid=sid, bilder=n, ok=ok, rc=r.returncode, ende=(r.stderr or "")[-600:] if not ok else ""))
        log(sid, n, "OK" if ok else "FEHLER", (r.stderr or "")[-250:].replace("\n", " ") if not ok else "")
        json.dump(INFO, open(OUT + "info.json", "w"), indent=1)
        if ok:
            break

ths = [threading.Thread(target=leiter, args=("rec2125_seq012", 0, (100, 50, 25))), threading.Thread(target=leiter, args=("rec2123_seq065", 1, (50, 25, 12)))]
[t.start() for t in ths]; [t.join() for t in ths]
log("FERTIG")
