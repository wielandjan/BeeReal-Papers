# BeeReal Paper 1, Versuch 05. Code genau wie in versuche/05_tracking_bildrate/notebooks/notebook_bildrate.ipynb ausgeführt.
# BeeReal Bildraten-Test: Bienensuche (Grounding DINO) auf den ersten 10 s zweier Val-Sequenzen mit jedem 2. Bild
# (ca. 30 Bilder/s). Daraus lassen sich lokal 30, 15 und 10 Bilder/s ableiten (jedes 2., 4., 6. Bild).
# Frage: Werden Tracks besser, wenn mehr Bilder pro Sekunde ausgewertet werden?
CLIPS = {"rec2123_seq065": ("gelb", (90, 800, 1080, 1025), 100), "rec2125_seq012": ("rot", (0, 750, 1080, 1035), 150)}
FRAMES = range(0, 600, 2)      # 10 s, jedes 2. Bild bei 59,94 fps
THR = 0.15
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
import gzip, json, threading, time
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

START = time.time(); OUT = "/kaggle/working/"
def log(*a):
    print(f"[{(time.time()-START)/60:6.1f} min]", *a, flush=True)

GD_ID = "IDEA-Research/grounding-dino-base"
gproc = AutoProcessor.from_pretrained(GD_ID)
N_GPU = max(torch.cuda.device_count(), 1)
gmodels = [AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(f"cuda:{i}").eval() for i in range(N_GPU)]
log("Grounding DINO auf", N_GPU, "GPU(s)")

# Identisch zum Baseline-Skript (find_bees / detect_tiled)
def find_bees(img, g, thr=THR):
    inp = gproc(images=img, text="a bee.", return_tensors="pt").to(f"cuda:{g}")
    with torch.no_grad():
        out = gmodels[g](**inp)
    r = gproc.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=thr,
                                                     target_sizes=[img.size[::-1]])[0]
    return r["boxes"].cpu().tolist(), r["scores"].cpu().tolist()

def detect_tiled(img, tile, overlap, g, max_side):
    W, Hh = img.size; bs, ss = [], []
    for y in range(0, max(Hh - overlap, 1), tile - overlap):
        for x in range(0, max(W - overlap, 1), tile - overlap):
            b, s = find_bees(img.crop((x, y, min(x + tile, W), min(y + tile, Hh))), g)
            for (x1, y1, x2, y2), sc in zip(b, s):
                if 15 < x2 - x1 < max_side and 15 < y2 - y1 < max_side:
                    bs.append([x1 + x, y1 + y, x2 + x, y2 + y]); ss.append(sc)
    if not bs:
        return []
    B, S = torch.tensor(bs), torch.tensor(ss)
    return B[nms(B, S, 0.3)].tolist()

def run(sid, g):
    hive, roi, max_side = CLIPS[sid]
    dec = VideoDecoder(hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset"))
    H = roi[3] - roi[1]; tile = max(2 * H, 300); overlap = max(H // 2, 75); dets = {}
    for k, f in enumerate(FRAMES):
        board = Image.fromarray(dec.get_frame_at(f).data.permute(1, 2, 0).numpy()).crop(roi)
        dets[str(f)] = detect_tiled(board, tile, overlap, g, max_side)
        if k % 25 == 0:
            log(sid, "Bild", f, "Bienen:", len(dets[str(f)]))
    with gzip.open(OUT + f"dets_{sid}_fps30.json.gz", "wt") as fh:
        json.dump(dets, fh)
    log(sid, "fertig")

ths = [threading.Thread(target=run, args=(sid, i % N_GPU)) for i, sid in enumerate(CLIPS)]
[t.start() for t in ths]; [t.join() for t in ths]
log("FERTIG")
