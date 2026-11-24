# BeeReal Paper 1, Versuch 07 (SAM 2.1 Tiny als Videotracker).
# BeeReal Tracker-Vergleich: SAM 2.1 Tiny (Video-Tracker mit Gedächtnis, klein) auf den 6 Val-Ausschnitten (15 s, 5 Bilder/s).
# Vorgehen in Fenstern: Am Anfang jedes Fensters (alle 15 Bilder = 3 s) werden die sicheren SAM-3-Kästen (>= 0,7) als Startkästen
# gegeben, SAM 2.1 verfolgt jede Biene durch das Fenster (auch wenn der Detektor sie verliert). Benachbarte Fenster teilen ein Bild;
# dort werden die Objekte über IoU (Hungarian, >= 0,5) zu durchgehenden Tracks verbunden.
# Ausgabe pro Ausschnitt: Track -> Liste [bild, x1, y1, x2, y2] im ROI-Ausschnitt. Vergleich mit ByteTrack erst gegen die Val-Ground-Truth.
CLIPS = ["rec2123_seq021", "rec2123_seq036", "rec2123_seq073", "rec2125_seq001", "rec2125_seq005", "rec2125_seq012"]
STEP, FENSTER, START_MIN = 12, 15, 0.7
MODELL = "facebook/sam2.1-hiera-tiny"
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
WORKER = r'''
import glob, json, os, sys, time, traceback
import numpy as np, torch
from PIL import Image
from huggingface_hub import hf_hub_download
from scipy.optimize import linear_sum_assignment
from torchcodec.decoders import VideoDecoder
from transformers import Sam2VideoModel, Sam2VideoProcessor
OUT, STEP, FENSTER, START_MIN, MODELL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]; sids = sys.argv[6:]
ROI = {"2123": (90, 800, 1080, 1025), "2125": (0, 750, 1080, 1035)}
proc = Sam2VideoProcessor.from_pretrained(MODELL)
model = Sam2VideoModel.from_pretrained(MODELL, dtype=torch.float16).to("cuda:0").eval()

def iou(a, b):
    a = np.array(a, float).reshape(-1, 4); b = np.array(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return i / (ar(a)[:, None] + ar(b)[None] - i + 1e-9)

def fenster(ims, boxes):
    """verfolgt die Startkästen durch die Bilder ims; gibt {objekt: {lokales_bild: kasten}}"""
    sess = proc.init_video_session(video=ims, inference_device="cuda:0", video_storage_device="cpu", processing_device="cpu", dtype=torch.float16)
    ids = list(range(1, len(boxes) + 1))
    if not ids:
        return {}
    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=ids, input_boxes=[[list(map(float, b)) for b in boxes]])
    res = {i: {} for i in ids}
    with torch.no_grad():
        model(inference_session=sess, frame_idx=0)
        for o in model.propagate_in_video_iterator(inference_session=sess, start_frame_idx=0):
            m = proc.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]], binarize=True)[0]
            m = m[:, 0] if m.dim() == 4 else m
            oid = list(o.object_ids) if getattr(o, "object_ids", None) is not None else ids
            for k, i in enumerate(oid):
                ys, xs = torch.nonzero(m[k], as_tuple=True)
                if len(xs) >= 20:
                    res[int(i)][int(o.frame_idx)] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return res

for sid in sids:
    try:
        t0 = time.time(); fps = 59.94
        dec = VideoDecoder(hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset"))
        bild = json.load(open(glob.glob(f"/kaggle/input/**/bild_{sid}_0-15s.json", recursive=True)[0]))
        frames = sorted(int(f) for f in bild)
        ims = [Image.fromarray(dec.get_frame_at(f).data.permute(1, 2, 0).numpy()).crop(ROI[sid[3:7]]) for f in frames]
        tracks, offen, gid = {}, {}, 0          # offen: Objekt im letzten Fenster -> globale ID
        for s in range(0, len(frames) - 1, FENSTER):
            e = min(s + FENSTER, len(frames) - 1)
            start = [r[1:5] for r in bild[str(frames[s])] if r[0] >= START_MIN]
            res = fenster(ims[s:e + 1], start)
            # Verbinden: Startkästen dieses Fensters gegen Kästen der offenen Objekte im selben Bild
            neu = {}
            if offen:
                alt_ids = list(offen); alt_box = [offen[a] for a in alt_ids]
                if alt_box and start:
                    M = iou(alt_box, start); r, c = linear_sum_assignment(-M)
                    for i, j in zip(r, c):
                        if M[i, j] >= 0.5:
                            neu[j + 1] = alt_ids[i]
            offen = {}
            for oid, verlauf in res.items():
                g = neu.get(oid)
                if g is None:
                    g = gid; gid += 1
                for k, b in sorted(verlauf.items()):
                    if k == 0 and neu.get(oid) is not None:
                        continue   # gemeinsames Bild nicht doppelt
                    tracks.setdefault(g, []).append([frames[s + k]] + [round(x, 1) for x in b])
                if e - s in verlauf:
                    offen[g] = verlauf[e - s]
        json.dump({str(k): v for k, v in tracks.items()}, open(os.path.join(OUT, f"sam2_{sid}.json"), "w"))
        L = [(v[-1][0] - v[0][0]) / fps for v in tracks.values()]
        print("OK", sid, len(tracks), "Tracks, Median", round(float(np.median(L)), 2), "s,", round(time.time() - t0), "s,", round(torch.cuda.max_memory_allocated() / 1e9, 1), "GB", flush=True)
    except Exception:
        print("FEHLER", sid, traceback.format_exc()[-900:], flush=True)
'''
# %%
import os, subprocess, sys, threading, time
START = time.time(); OUT = "/kaggle/working/"
open(OUT + "worker.py", "w").write(WORKER)
teile = [CLIPS[0::2], CLIPS[1::2]]
def lauf(g, s):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    r = subprocess.run([sys.executable, OUT + "worker.py", OUT, str(STEP), str(FENSTER), str(START_MIN), MODELL] + s, env=env, capture_output=True, text=True)
    print(f"GPU {g}:", r.stdout, (r.stderr or "")[-1500:] if r.returncode else "", flush=True)
ths = [threading.Thread(target=lauf, args=(g, t)) for g, t in enumerate(teile)]
[t.start() for t in ths]; [t.join() for t in ths]
print("FERTIG", round((time.time() - START) / 60, 1), "min", sorted(f for f in os.listdir(OUT) if f.startswith("sam2_")), flush=True)
