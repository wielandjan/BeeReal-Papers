# BeeReal Paper 1, Versuch 01 (Grounding DINO, Positions-Tracker). Aktueller Stand, das Notebook von Lauf 1 ist nicht erhalten. Die lokalen Skripte nutzen daraus die Funktion track().
# BeeReal Baseline: Detektion -> Tracking -> Crops -> Re-ID (Zero-Shot und Training) -> Bericht
# Test-Sequenzen werden absichtlich NICHT verarbeitet. Ausgewertet wird nur auf Val.
SMOKE = False  # True: kurzer Probelauf, False: kompletter Lauf
# Lauf 2: Detektionen aus Lauf 1 wiederverwenden (Datensatz beereal-dets), SAM 3 auslassen,
# und im Training nur Tracks in einen Batch nehmen, die zur selben Zeit im Bild sind.
RUN_SAM3 = False
# Lauf 3: Grundmodelle vergleichen, jeweils ohne (Baseline) und mit unserem Training
ZERO_SHOT = ["dinov2_small", "dinov3_small", "dinov3_base", "mega_tiny", "mega_base", "mega_large"]
TRAIN = ["dinov2_small", "mega_tiny", "dinov3_small", "mega_base", "dinov3_base"]
RUN_TITLE = "Lauf 3: Grundmodelle im Vergleich"
# True: abbrechen, wenn gespeicherte Detektionen fehlen, statt 4 h Grounding DINO neu zu rechnen (spart GPU-Budget)
ONLY_OLD_DETS = False
# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "datasets", "torchcodec", "transformers", "timm"], check=False)
# %%
import glob, re
from huggingface_hub import login
_f = glob.glob("/kaggle/input/**/hf_token*", recursive=True)
if not _f:
    _f = [p for p in glob.glob("/kaggle/input/**/*", recursive=True) if p.endswith((".txt", ".csv"))]
login(re.search(r"hf_[A-Za-z0-9]+", open(_f[0], encoding="utf-8-sig").read()).group(0))
print("HF_LOGIN OK", flush=True)
# %%
import copy, gc, gzip, io, json, os, pickle, queue, random, threading, time, traceback
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as pt
from PIL import Image, ImageOps
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
from torchvision import transforms as T
from torchvision.ops import nms, box_iou
from transformers import AutoProcessor, AutoModel, AutoModelForZeroShotObjectDetection

START = time.time()
OUT, TMP = "/kaggle/working/", "/kaggle/temp/"
os.makedirs(TMP, exist_ok=True)
FPS, STEP = 59.94, 6
MAX_GAP, REL_DIST, MIN_CROPS, MAX_CROPS = 5, 0.5, 10, 40
HOURS_DETECT, HOURS_TOTAL = 7.0, 10.5
HIVE = {2123: "gelb", 2125: "rot"}
ROI_YELLOW = (90, 800, 1080, 1025)
ROI_RED = (0, 750, 1080, 1035)
START_FRAME = {"rec2125_seq000": 900}   # Kamera-Aufbau: erst ab Sekunde 15 verwenden
SPLIT = {2123: [("train", 15, 59), ("luecke", 60, 64), ("val", 65, 74), ("luecke", 75, 79), ("test", 80, 89)],
         2125: [("train", 0, 9), ("luecke", 10, 11), ("val", 12, 14), ("luecke", 15, 15), ("test", 16, 20)]}

if SMOKE:
    N_FRAMES, GAP_FRAMES, DRAWS = 240, 120, 3
    WANTED = {(2123, 15), (2123, 17), (2123, 65), (2125, 1), (2125, 12)}
    STEPS_HEAD, STEPS_FT, EVAL_EVERY = 20, 20, 10
else:
    N_FRAMES, GAP_FRAMES, DRAWS = None, 600, 20      # Mindestabstand 600 Frames = ca. 10 s
    WANTED = {(2123, i) for i in range(15, 60, 2)} | {(2123, i) for i in range(65, 75)} \
           | {(2125, i) for i in range(0, 10)} | {(2125, i) for i in range(12, 15)}
    STEPS_HEAD, STEPS_FT, EVAL_EVERY = 1000, 2000, 500

REPORT = {"smoke": SMOKE, "errors": [], "config": dict(MAX_GAP=MAX_GAP, REL_DIST=REL_DIST, MIN_CROPS=MIN_CROPS,
          MAX_CROPS=MAX_CROPS, GAP_FRAMES=GAP_FRAMES, n_sequences=len(WANTED))}

def log(*a):
    print(f"[{(time.time()-START)/60:6.1f} min]", *a, flush=True)

def stage(name, fn):
    log("START", name)
    try:
        r = fn(); log("ENDE", name); return r
    except Exception as e:
        REPORT["errors"].append({"stage": name, "error": repr(e), "trace": traceback.format_exc()[-1500:]})
        log("FEHLER", name, repr(e)); traceback.print_exc(); return None

def save_report():
    json.dump(REPORT, open(OUT + "results.json", "w"), indent=1, default=str)

def split_of(rec, idx):
    return next((n for n, a, b in SPLIT[rec] if a <= idx <= b), "FEHLER")

def get_frame(video, i):
    return Image.fromarray(video.get_frame_at(i).data.permute(1, 2, 0).numpy())
# %%
# SAM 3 als Vergleich an drei Frames. Darf scheitern (z. B. fehlender Zugang), der Lauf geht dann weiter.
ds = load_dataset("mrkschtr/thehiveisreal", split="train", streaming=True)
sample0 = next(iter(ds))
log("Erste Sequenz:", sample0["sequence_id"])

def sam3_test():
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda:0").eval()
    res = []
    for f in (0, 30, 60) if SMOKE else (0, 600, 1200):
        board = get_frame(sample0["video"], f).crop(ROI_YELLOW)
        t0 = time.time()
        inp = proc(images=board, text="bee", return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                    target_sizes=inp.get("original_sizes").tolist())[0]
        boxes = r["boxes"].cpu().tolist()
        res.append(dict(frame=f, n=len(boxes), seconds=round(time.time() - t0, 2),
                        gpu_gb=round(torch.cuda.max_memory_allocated(0) / 1e9, 2)))
        fig, ax = plt.subplots(figsize=(20, 6)); ax.imshow(board); ax.axis("off")
        ax.set_title(f"SAM 3, Frame {f}: {len(boxes)} Objekte")
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(pt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color="lime", lw=1.2))
        plt.savefig(OUT + f"sam3_frame{f}.png", dpi=90, bbox_inches="tight"); plt.close()
    del model; gc.collect(); torch.cuda.empty_cache()
    return res

if RUN_SAM3:
    REPORT["sam3"] = stage("SAM 3 Vergleich", sam3_test)
save_report()
# %%
# Grounding DINO auf allen GPUs
GD_ID = "IDEA-Research/grounding-dino-base"
gproc = AutoProcessor.from_pretrained(GD_ID)
N_GPU = max(torch.cuda.device_count(), 1)
# Fertige Detektionen aus einem früheren Lauf (Datensatz beereal-dets). Fehlt eine, wird sie neu berechnet.
# Kaggle entpackt .gz-Dateien in Datensätzen eventuell, deshalb beide Endungen suchen.
OLD_DETS = {re.search(r"dets_(rec\d+_seq\d+)", p).group(1): p
            for p in glob.glob("/kaggle/input/**/dets_*.json*", recursive=True)}
print("Eingaben:", glob.glob("/kaggle/input/*/*")[:5], flush=True)
NEED_GD = any(f"rec{r}_seq{i:03d}" not in OLD_DETS for r, i in WANTED)
log("Fertige Detektionen:", len(OLD_DETS), "Grounding DINO nötig:", NEED_GD)
if NEED_GD and ONLY_OLD_DETS:
    raise RuntimeError("Gespeicherte Detektionen fehlen – Abbruch, damit keine 4 h GPU-Zeit verbraucht werden")
gmodels = [AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(f"cuda:{i}").eval() for i in range(N_GPU)] if NEED_GD else []
glocks = [threading.Lock() for _ in range(N_GPU)]
log("Grounding DINO auf", len(gmodels), "GPU(s)")

def find_bees(img, g, thr=0.15):
    dev = f"cuda:{g}"
    inp = gproc(images=img, text="a bee.", return_tensors="pt").to(dev)
    with glocks[g], torch.no_grad():
        out = gmodels[g](**inp)
    r = gproc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=thr, text_threshold=thr, target_sizes=[img.size[::-1]])[0]
    return r["boxes"].cpu().tolist(), r["scores"].cpu().tolist()

def detect_tiled(img, tile, overlap, g, max_side, thr=0.15):
    W, Hh = img.size
    bs, ss = [], []
    for y in range(0, max(Hh - overlap, 1), tile - overlap):
        for x in range(0, max(W - overlap, 1), tile - overlap):
            b, s = find_bees(img.crop((x, y, min(x + tile, W), min(y + tile, Hh))), g, thr)
            for (x1, y1, x2, y2), sc in zip(b, s):
                if 15 < x2 - x1 < max_side and 15 < y2 - y1 < max_side:
                    bs.append([x1 + x, y1 + y, x2 + x, y2 + y]); ss.append(sc)
    if not bs:
        return []
    B, S = torch.tensor(bs), torch.tensor(ss)
    return B[nms(B, S, 0.3)].tolist()

def estimate_roi(sample):
    """Automatisch geschätzter Brett-Ausschnitt aus Detektionen im ganzen Bild. Muss von Hand geprüft werden."""
    n = sample["n_frames"]
    cen, sizes = [], []
    for f in np.linspace(0, n - 1, 5).astype(int).tolist():
        img = get_frame(sample["video"], f)
        for x1, y1, x2, y2 in detect_tiled(img, 540, 135, 0, 200):
            cen.append(((x1 + x2) / 2, (y1 + y2) / 2)); sizes.append(max(x2 - x1, y2 - y1))
    cen = np.array(cen); W, Hh = sample["width"], sample["height"]
    x1, y1 = np.percentile(cen, 2, axis=0); x2, y2 = np.percentile(cen, 98, axis=0)
    size = float(np.median(sizes)); m = 1.5 * size
    roi = (max(int(x1 - m), 0), max(int(y1 - m), 0), min(int(x2 + m), W), min(int(y2 + m), Hh))
    return roi, size, len(cen)

def save_roi_image(sample, roi, name):
    img = get_frame(sample["video"], 0)
    fig, ax = plt.subplots(figsize=(8, 14)); ax.imshow(img); ax.axis("off")
    ax.add_patch(pt.Rectangle(roi[:2], roi[2]-roi[0], roi[3]-roi[1], fill=False, color="cyan", lw=2))
    ax.set_title(f"{name}: ROI {roi}")
    plt.savefig(OUT + f"roi_{name}.png", dpi=70, bbox_inches="tight"); plt.close()
# %%
def track(dets, max_gap=MAX_GAP, rel_dist=REL_DIST):
    tracks, active, nid = {}, {}, 0
    for k, f in enumerate(sorted(dets)):
        boxes = dets[f]
        cen = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes]).reshape(-1, 2)
        size = np.array([max(b[2]-b[0], b[3]-b[1]) for b in boxes])
        used, ids = set(), list(active)
        if ids and len(cen):
            pc = np.array([active[i][0] for i in ids]); ps = np.array([active[i][1] for i in ids])
            dist = np.linalg.norm(pc[:, None] - cen[None], axis=2)
            cost = np.where(dist < rel_dist * np.minimum(ps[:, None], size[None]), dist, 1e6)
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e6:
                    tracks[ids[i]].append((f, boxes[j])); active[ids[i]] = (cen[j], size[j], k); used.add(j)
        for j in range(len(boxes)):
            if j not in used:
                tracks[nid] = [(f, boxes[j])]; active[nid] = (cen[j], size[j], k); nid += 1
        active = {i: v for i, v in active.items() if k - v[2] <= max_gap}
    return tracks

def remove_spikes(t):
    """Entfernt einzelne Ausreißer wie den Sprung auf eine überfliegende Biene."""
    if len(t) < 3:
        return t
    c = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for _, b in t])
    s = np.median([max(b[2]-b[0], b[3]-b[1]) for _, b in t]) * 0.3
    keep = [True] * len(t)
    for i in range(1, len(t) - 1):
        if np.linalg.norm(c[i]-c[i-1]) > s and np.linalg.norm(c[i]-c[i+1]) > s and np.linalg.norm(c[i-1]-c[i+1]) < s:
            keep[i] = False
    return [x for x, k in zip(t, keep) if k]

def build_sequence(sample, roi, max_side, g):
    video, sid = sample["video"], sample["sequence_id"]
    n = N_FRAMES or sample["n_frames"]
    H = roi[3] - roi[1]
    tile = max(2 * H, 300); overlap = max(H // 2, 75)
    dets, boards = {}, {}
    if sid in OLD_DETS:
        with (gzip.open if OLD_DETS[sid].endswith(".gz") else open)(OLD_DETS[sid], "rt") as fh:
            dets = {int(k): v for k, v in json.load(fh).items()}
        board = lambda f: boards[f] if f in boards else boards.setdefault(f, get_frame(video, f).crop(roi))
    else:
        for f in range(START_FRAME.get(sid, 0), n, STEP):
            boards[f] = get_frame(video, f).crop(roi)
            dets[f] = detect_tiled(boards[f], tile, overlap, g, max_side)
        with gzip.open(OUT + f"dets_{sid}.json.gz", "wt") as fh:
            json.dump({str(k): v for k, v in dets.items()}, fh)
        board = lambda f: boards[f]
    tracks = track(dets)
    # Crops, die einen anderen Kasten stark überlappen, fallen weg (Ersatz für Chans Dichte-Filter)
    keep_box = {}
    for f, b in dets.items():
        if len(b) > 1:
            iou = box_iou(torch.tensor(b), torch.tensor(b)); iou.fill_diagonal_(0)
            keep_box[f] = (iou.max(dim=1).values < 0.3).tolist()
        else:
            keep_box[f] = [True] * len(b)
    idx_of = {f: {tuple(x): i for i, x in enumerate(b)} for f, b in dets.items()}
    L = np.array([len(t) for t in tracks.values()]) if tracks else np.array([0])
    out, n_filtered = {}, 0
    for tid, t in tracks.items():
        t2 = [(f, b) for f, b in remove_spikes(t) if keep_box[f][idx_of[f][tuple(b)]]]
        n_filtered += len(t) - len(t2)
        if len(t2) < MIN_CROPS:
            continue
        c = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for _, b in t2])
        size = float(np.median([max(b[2]-b[0], b[3]-b[1]) for _, b in t2]))
        moved = float(np.linalg.norm(c.max(0) - c.min(0))) > size
        crops = []
        for k in np.unique(np.linspace(0, len(t2) - 1, MAX_CROPS).astype(int)):
            f, b = t2[k]
            buf = io.BytesIO(); board(f).crop(tuple(map(int, b))).save(buf, "JPEG", quality=95)
            crops.append((f, buf.getvalue()))
        out[tid] = dict(crops=crops, moved=moved, first=t2[0][0], last=t2[-1][0])
    stat = dict(sequence_id=sid, recording=sample["source_recording"], index=sample["sequence_index"],
                split=split_of(sample["source_recording"], sample["sequence_index"]),
                bees_per_frame=round(float(np.mean([len(b) for b in dets.values()])), 1), tracks_all=len(tracks),
                tracks_kept=len(out), tracks_moved=sum(v["moved"] for v in out.values()),
                median_len=float(np.median(L)), crops_filtered=n_filtered)
    pickle.dump(dict(stat=stat, tracks=out), open(TMP + f"{sid}.pkl", "wb"))
    return stat

ROI, MAXSIDE = {2123: ROI_YELLOW, 2125: ROI_RED}, {2123: 100, 2125: 150}
STATS, SEQ_ROWS, SEEN = [], [], set()

def run_detection():
    q = queue.Queue(maxsize=N_GPU)
    def worker(g):
        while True:
            s = q.get()
            if s is None:
                return
            t0 = time.time()
            try:
                st = build_sequence(s, ROI[s["source_recording"]], MAXSIDE[s["source_recording"]], g)
                STATS.append(st); log(f"GPU{g}", st["sequence_id"], st["split"], "Bienen/Frame", st["bees_per_frame"],
                                      "Tracks", st["tracks_kept"], f"{time.time()-t0:.0f}s")
            except Exception as e:
                REPORT["errors"].append({"stage": "Sequenz " + s["sequence_id"], "error": repr(e),
                                         "trace": traceback.format_exc()[-1500:]})
                log("FEHLER Sequenz", s["sequence_id"], repr(e))
    ths = [threading.Thread(target=worker, args=(g,)) for g in range(N_GPU)]
    [t.start() for t in ths]
    done = 0
    for s in load_dataset("mrkschtr/thehiveisreal", split="train", streaming=True):
        rec, idx = s["source_recording"], s["sequence_index"]
        SEQ_ROWS.append(dict(sequence_id=s["sequence_id"], recording=rec, index=idx, start_time_s=s["start_time_s"],
                             split=split_of(rec, idx)))
        if (rec, idx) not in WANTED:
            continue
        if (time.time() - START) / 3600 > HOURS_DETECT:
            log("Zeitlimit Detektion erreicht, überspringe", s["sequence_id"]); continue
        if rec not in SEEN:
            SEEN.add(rec); save_roi_image(s, ROI[rec], HIVE[rec])
        q.put(s); done += 1
        if SMOKE and done == len(WANTED):
            break
    for _ in ths:
        q.put(None)
    [t.join() for t in ths]
    df = pd.DataFrame(SEQ_ROWS).sort_values(["recording", "index"])
    df.to_csv(OUT + "split.csv", index=False)
    REPORT["start_time_diffs"] = {str(k): v.start_time_s.diff().value_counts().to_dict() for k, v in df.groupby("recording")}
    REPORT["sequence_stats"] = STATS
    REPORT["roi_used"] = {str(k): v for k, v in ROI.items()}
    pd.DataFrame(STATS).to_csv(OUT + "sequence_stats.csv", index=False)

stage("Detektion und Tracking", run_detection)
save_report()
del gmodels; gc.collect(); torch.cuda.empty_cache()
# %%
# Alle Crops laden
JPEG, SEQS = [], {}
for p in sorted(glob.glob(TMP + "*.pkl")):
    d = pickle.load(open(p, "rb")); st = d["stat"]; trs = {}
    for tid, t in d["tracks"].items():
        ids = []
        for f, jb in t["crops"]:
            JPEG.append(jb); ids.append((f, len(JPEG) - 1))
        trs[tid] = dict(crops=ids, moved=t["moved"], first=t["first"], last=t["last"])
    SEQS[st["sequence_id"]] = dict(hive=HIVE[st["recording"]], split=st["split"], tracks=trs)
REPORT["data"] = {f"{h}_{s}": dict(seqs=sum(v["hive"] == h and v["split"] == s for v in SEQS.values()),
                  tracks=sum(len(v["tracks"]) for v in SEQS.values() if v["hive"] == h and v["split"] == s))
                  for h in HIVE.values() for s in ("train", "val")}
log("Crops:", len(JPEG), REPORT["data"]); save_report()

def track_sheet(hive, n=6):
    pool = [(v, t) for v in SEQS.values() if v["hive"] == hive for t in v["tracks"].values()]
    random.Random(0).shuffle(pool); pool = pool[:n]
    if not pool:
        return
    fig, axes = plt.subplots(len(pool), 10, figsize=(12, 1.4 * len(pool)), squeeze=False)
    for row, (_, t) in zip(axes, pool):
        for ax_, k in zip(row, np.linspace(0, len(t["crops"]) - 1, 10).astype(int)):
            ax_.imshow(Image.open(io.BytesIO(JPEG[t["crops"][k][1]]))); ax_.axis("off")
    plt.tight_layout(); plt.savefig(OUT + f"tracks_{hive}.png", dpi=80); plt.close()
for h in HIVE.values():
    stage("Track-Bilder " + h, lambda h=h: track_sheet(h))
# %%
# Auswertung nach Chan: Galerie aus 1 richtigen Bild + 10 Distraktoren, Query und Treffer mit Mindestabstand.
# Wahrheit sind automatische Tracks, nicht von Hand bestätigt.
def overlap(a, b):
    return a["first"] <= b["last"] and b["first"] <= a["last"]

def plan_eval(seq_ids, draws, seed=0, max_tracks=None):
    rng = random.Random(seed); trials = []
    for sid in seq_ids:
        trs = SEQS[sid]["tracks"]; ids = list(trs); rng.shuffle(ids); used = 0
        for tid in ids:
            t = trs[tid]
            if t["last"] - t["first"] < GAP_FRAMES:
                continue
            others = [o for o in trs if o != tid and overlap(t, trs[o])]
            if len(others) < 10:
                continue
            qs = [c for c in t["crops"] if c[0] <= t["last"] - GAP_FRAMES]
            for _ in range(draws):
                q = rng.choice(qs); pos = [c[1] for c in t["crops"] if c[0] >= q[0] + GAP_FRAMES]
                if not pos:
                    continue
                dist = [rng.choice(trs[o]["crops"])[1] for o in rng.sample(others, 10)]
                trials.append(dict(hive=SEQS[sid]["hive"], track=f"{sid}/{tid}", moved=t["moved"], q=q[1], pos=rng.choice(pos),
                                   dist=dist, allpos=pos, alldist=[c[1] for o in others for c in trs[o]["crops"][::8]]))
            used += 1
            if max_tracks and used >= max_tracks:
                break
    return trials

EVAL_TF = T.Compose([T.Lambda(lambda im: ImageOps.pad(im, (224, 224), color=(0, 0, 0))), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
class Crops(torch.utils.data.Dataset):
    def __init__(self, idx, tf): self.idx, self.tf = idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.tf(Image.open(io.BytesIO(JPEG[self.idx[i]])).convert("RGB"))

def embed_with(fn, idx, bs=128):
    dl = torch.utils.data.DataLoader(Crops(idx, EVAL_TF), batch_size=bs, num_workers=2)
    with torch.no_grad():
        out = [torch.nn.functional.normalize(fn(x), dim=1).cpu() for x in dl]
    return {i: r for i, r in zip(idx, torch.cat(out))}

def evaluate(fn, trials, boot=500):
    need = sorted({i for t in trials for i in [t["q"], t["pos"], *t["dist"], *t["allpos"], *t["alldist"]]})
    E = embed_with(fn, need)
    rows = []
    for t in trials:
        q = E[t["q"]]
        s_pos = float(q @ E[t["pos"]]); s_d = torch.stack([E[i] for i in t["dist"]]) @ q
        rank = 1 + int((s_d >= s_pos).sum())
        P = torch.stack([E[i] for i in t["allpos"]]) @ q; N = torch.stack([E[i] for i in t["alldist"]]) @ q
        sc = torch.cat([P, N]); lab = torch.cat([torch.ones(len(P)), torch.zeros(len(N))])
        hits = lab[sc.argsort(descending=True)]
        ap = float((hits.cumsum(0) / torch.arange(1, len(hits) + 1))[hits == 1].mean())
        rows.append(dict(hive=t["hive"], track=t["track"], moved=t["moved"], rank=rank, ap=ap, gal=len(hits)))
    df = pd.DataFrame(rows); res = {}
    if df.empty:
        return res
    for hive in df.hive.unique():
        for name, sub in (("alle", df[df.hive == hive]), ("bewegt", df[(df.hive == hive) & df.moved])):
            if sub.empty:
                continue
            per = sub.assign(r1=sub["rank"] == 1).groupby("track").r1.mean().values
            rng = np.random.default_rng(0)
            b = [rng.choice(per, len(per)).mean() for _ in range(boot)]
            res[f"{hive}_{name}"] = dict(tracks=int(len(per)), rank1=round(float((sub["rank"] == 1).mean()), 3),
                rank1_ci=[round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)],
                rank3=round(float((sub["rank"] <= 3).mean()), 3), rank5=round(float((sub["rank"] <= 5).mean()), 3),
                mAP=round(float(sub.ap.mean()), 3), gallery_mean=round(float(sub.gal.mean()), 1))
    return res

VAL = [s for s, v in SEQS.items() if v["split"] == "val"]
TRIALS = plan_eval(VAL, DRAWS)
QUICK = plan_eval(VAL, 3, seed=1, max_tracks=15)
log("Val-Versuche:", len(TRIALS), "schnell:", len(QUICK))
RES = REPORT.setdefault("results", {})

def pixel_fn(x):
    return torch.nn.functional.interpolate(x, size=32, mode="bilinear").flatten(1)

# Grundmodelle (Backbones): Quelle, Name auf Hugging Face, Bildgröße, Normierung.
# Die Bilder kommen immer als 224 px mit ImageNet-Normierung und werden pro Modell umgerechnet.
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]); HALF = ([0.5] * 3, [0.5] * 3)
BACKBONES = {
    "dinov2_small": ("hf", "facebook/dinov2-small", 224, IMNET),
    "dinov3_small": ("hf", "facebook/dinov3-vits16-pretrain-lvd1689m", 224, IMNET),
    "dinov3_base":  ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m", 224, IMNET),
    "dinov3_large": ("hf", "facebook/dinov3-vitl16-pretrain-lvd1689m", 224, IMNET),
    "dinov3_hplus": ("hf", "facebook/dinov3-vith16plus-pretrain-lvd1689m", 224, IMNET),
    "mega_tiny":    ("timm", "hf-hub:BVRA/MegaDescriptor-T-224", 224, HALF),
    "mega_base":    ("timm", "hf-hub:BVRA/MegaDescriptor-B-224", 224, HALF),
    "mega_large":   ("timm", "hf-hub:BVRA/MegaDescriptor-L-384", 384, HALF),
}

class Net(torch.nn.Module):
    """Einheitliche Hülle: Bild rein, Merkmalsvektor raus, egal ob DINO (transformers) oder MegaDescriptor (timm)."""
    def __init__(self, key):
        super().__init__()
        self.src, name, self.size, (mean, std) = BACKBONES[key]
        if self.src == "hf":
            self.m = AutoModel.from_pretrained(name); self.dim = self.m.config.hidden_size
        else:
            import timm
            self.m = timm.create_model(name, pretrained=True, num_classes=0); self.dim = self.m.num_features
        self.register_buffer("a", torch.tensor(IMNET[0]).view(1, 3, 1, 1)); self.register_buffer("b", torch.tensor(IMNET[1]).view(1, 3, 1, 1))
        self.register_buffer("c", torch.tensor(mean).view(1, 3, 1, 1)); self.register_buffer("d", torch.tensor(std).view(1, 3, 1, 1))
    def blocks(self):
        m = self.m
        if self.src == "timm":
            return [b for stage_ in m.layers for b in stage_.blocks]
        if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
            return list(m.encoder.layer)
        if hasattr(m, "layer"):
            return list(m.layer)
        return list(max((x for x in m.modules() if isinstance(x, torch.nn.ModuleList)), key=len))
    def forward(self, x):
        x = x * self.b + self.a
        if self.size != x.shape[-1]:
            x = torch.nn.functional.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.c) / self.d
        if self.src == "timm":
            return self.m(x)
        o = self.m(pixel_values=x)
        p = getattr(o, "pooler_output", None)
        return p if p is not None else o.last_hidden_state[:, 0]

def zero_shot(key):
    if (time.time() - START) / 3600 > HOURS_TOTAL:
        raise RuntimeError("Zeitlimit")
    bb = Net(key).to("cuda:0").eval()
    def fn(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return bb(x.to("cuda:0")).float()
    r = evaluate(fn, TRIALS); del bb; gc.collect(); torch.cuda.empty_cache()
    return r

RES["pixel_32x32"] = stage("Pixel-Vergleich", lambda: evaluate(pixel_fn, TRIALS)); save_report()
for key in ZERO_SHOT:
    if key == "mega_large":
        continue   # groß und langsam, läuft ganz am Ende
    RES[key + "_untrainiert"] = stage("Zero-Shot " + key, lambda k=key: zero_shot(k)); save_report()
# %%
# Training: Triplet Loss wie Chan (semi-hard, Margin 0,2). Ein Batch stammt aus einer Sequenz und damit aus einem Stock,
# zeitgleiche Tracks sind dann verschiedene Bienen, sofern der Tracker stimmt.
def semihard_triplet(e, y, margin=0.2):
    D = (2 - 2 * e @ e.T).clamp(min=1e-12).sqrt()
    same = y[:, None] == y[None]
    a, p = (same & ~torch.eye(len(y), dtype=torch.bool, device=y.device)).nonzero(as_tuple=True)
    dap, dan = D[a, p], D[a]
    semi = ~same[a] & (dan > dap[:, None]) & (dan < dap[:, None] + margin)
    has = semi.any(1)
    if not has.any():
        return e.sum() * 0
    hardest = dan.masked_fill(~semi, float("inf")).min(1).values
    return (dap[has] - hardest[has] + margin).clamp(min=0).mean()

TRAIN_TF = T.Compose([T.Lambda(lambda im: ImageOps.pad(im, (256, 256), color=(0, 0, 0))),
    T.RandomRotation(180), T.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.RandomGrayscale(0.1), T.RandomApply([T.GaussianBlur(3)], 0.3),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class TrainSet(torch.utils.data.Dataset):
    def __getitem__(self, i): return TRAIN_TF(Image.open(io.BytesIO(JPEG[i])).convert("RGB")), i
    def __len__(self): return len(JPEG)

class Sampler(torch.utils.data.Sampler):
    """zeitgleich=True: ein Batch enthält nur Tracks, die zu einem zufälligen Zeitpunkt alle im Bild sind.
    Dann sind es sicher verschiedene Bienen. Ohne das landet oft dieselbe Biene unter zwei Track-Nummern im Batch,
    weil der Tracker sie zwischendurch verliert, und das Training drückt sie fälschlich auseinander."""
    def __init__(self, hives, steps, P, K, seed=0, zeitgleich=False):
        self.steps, self.P, self.K, self.rng, self.zeitgleich = steps, P, K, random.Random(seed), zeitgleich
        self.by = {h: [s for s, v in SEQS.items() if v["split"] == "train" and v["hive"] == h and len(v["tracks"]) >= 2] for h in hives}
        self.by = {h: v for h, v in self.by.items() if v}
        self.label = {}
    def __len__(self): return self.steps
    def __iter__(self):
        for _ in range(self.steps):
            h = self.rng.choice(list(self.by)); sid = self.rng.choice(self.by[h]); trs = SEQS[sid]["tracks"]
            pool = list(trs)
            if self.zeitgleich:
                for _ in range(20):
                    t0 = trs[self.rng.choice(pool)]; at = self.rng.randint(t0["first"], t0["last"])
                    alive = [i for i in pool if trs[i]["first"] <= at <= trs[i]["last"]]
                    if len(alive) >= 8:
                        break
                pool = alive
            batch = []
            for n, tid in enumerate(self.rng.sample(pool, min(self.P, len(pool)))):
                for c in self.rng.choices(trs[tid]["crops"], k=self.K):
                    self.label[c[1]] = n; batch.append(c[1])
            yield batch

class ReID(torch.nn.Module):
    def __init__(self, key="dinov2_small"):
        super().__init__(); self.bb = Net(key)
        self.head = torch.nn.Linear(self.bb.dim, 128)
    def freeze(self, blocks=0):
        for p in self.bb.parameters(): p.requires_grad = False
        if blocks:
            for b in self.bb.blocks()[-blocks:]:
                for p in b.parameters(): p.requires_grad = True
            # Abschluss-Normierung des Grundmodells mit freigeben
            for n, p in self.bb.m.named_parameters():
                if n.split(".")[0] in ("norm", "layernorm", "fc_norm"):
                    p.requires_grad = True
    def forward(self, x):
        return torch.nn.functional.normalize(self.head(self.bb(x)), dim=1)

def model_fn(m):
    def fn(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return m(x.to("cuda:0")).float()
    return fn

def quick_score(m):
    m.eval(); r = evaluate(model_fn(m), QUICK, boot=1); m.train()
    v = [x["rank1"] for k, x in r.items() if k.endswith("_alle")]
    return float(np.mean(v)) if v else 0.0

def train(m, hives, steps, P, K, blocks, lr_head, lr_bb, name, zeitgleich=False):
    if (time.time() - START) / 3600 > HOURS_TOTAL:
        log("Zeitlimit, überspringe", name); return m
    m.to("cuda:0").train(); m.freeze(blocks)
    groups = [{"params": m.head.parameters(), "lr": lr_head}]
    if blocks:
        groups.append({"params": [p for p in m.bb.parameters() if p.requires_grad], "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda")
    sam = Sampler(hives, steps, P, K, zeitgleich=zeitgleich)
    dl = torch.utils.data.DataLoader(TrainSet(), batch_sampler=sam, num_workers=3, pin_memory=True)
    best, best_state, hist = -1, None, []
    for step, (x, idx) in enumerate(dl, 1):
        # Das Label ist der Track innerhalb des Batches. Es wird aus der Reihenfolge berechnet, weil der Sampler
        # in einem eigenen Prozess läuft und sein Wörterbuch hier nicht sichtbar ist.
        y = torch.arange(len(x)).div(K, rounding_mode="floor").to("cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            e = m(x.to("cuda:0", non_blocking=True))
        loss = semihard_triplet(e.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % EVAL_EVERY == 0 or step == steps:
            s = quick_score(m); hist.append(dict(step=step, loss=round(float(loss), 4), val_rank1=round(s, 3)))
            log(name, hist[-1], f"GPU {torch.cuda.max_memory_allocated(0)/1e9:.1f} GB")
            if s > best:
                best, best_state = s, copy.deepcopy({k: v.cpu() for k, v in m.state_dict().items()})
    m.load_state_dict(best_state); m.eval()
    REPORT.setdefault("training", {})[name] = dict(history=hist, best_quick_rank1=best,
        gpu_gb=round(torch.cuda.max_memory_allocated(0) / 1e9, 2))
    RES[name] = evaluate(model_fn(m), TRIALS)
    torch.save({k: v for k, v in m.state_dict().items() if k.startswith("head")}, OUT + f"{name}_head.pt")
    save_report()
    return m

def train_backbone(key):
    # Lauf 3: für jedes Grundmodell dasselbe Rezept wie der beste Stand aus Lauf 2:
    # erst nur Kopf, dann die letzten 4 Blöcke mit, zeitgleiche Pakete, beide Stöcke.
    both = ["gelb", "rot"]
    m = train(ReID(key), both, STEPS_HEAD, 32, 4, 0, 1e-3, 0, key + "_kopf", zeitgleich=True)
    m = train(m, both, STEPS_FT, 16, 4, 4, 1e-4, 1e-5, key + "_trainiert", zeitgleich=True)
    del m; gc.collect(); torch.cuda.empty_cache()

for key in TRAIN:
    stage("Training " + key, lambda k=key: train_backbone(k)); save_report()
if "mega_large" in ZERO_SHOT:
    RES["mega_large_untrainiert"] = stage("Zero-Shot mega_large", lambda: zero_shot("mega_large")); save_report()
# %%
# Bericht
def table(res):
    rows = []
    for model, r in res.items():
        for group, v in (r or {}).items():
            rows.append(dict(modell=model, gruppe=group, **{k: v[k] for k in ("tracks", "rank1", "rank3", "rank5", "mAP")},
                             ci=str(v["rank1_ci"])))
    return pd.DataFrame(rows)

tb = table(RES); tb.to_csv(OUT + "results.csv", index=False)
md = ["# BeeReal Baseline – " + RUN_TITLE, "",
      "Detektionen und Tracks wie Lauf 1/2 (Grounding DINO). Vergleich DINOv2, DINOv3, MegaDescriptor: `_untrainiert` = "
      "Baseline ohne unser Training, `_trainiert` = Kopf + letzte 4 Blöcke, zeitgleiche Pakete, beide Stöcke. "
      "Fehlt ein Modell, steht der Grund unter Fehler (z. B. kein Zugang).", "", f"Laufzeit: {(time.time()-START)/3600:.2f} h. Probelauf: {SMOKE}.", "",
      "Ausgewertet wird nur auf **Val**, Test wurde nicht verarbeitet. Die Wahrheit sind automatische Tracks, keine "
      "Handlabels. Rank-k: Galerie aus 1 richtigen Bild und 10 Distraktoren aus demselben Stock (Raten: Rank-1 = 9 %).",
      "", tb.to_string(index=False) if len(tb) else "Keine Ergebnisse.", "",
      "## Fehler", json.dumps(REPORT["errors"], indent=1)[:3000]]
open(OUT + "REPORT.md", "w", encoding="utf-8").write("\n".join(md))
save_report()
print("\n".join(md), flush=True)
log("FERTIG")
