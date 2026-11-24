# BeeReal Paper 1, Versuch 09. Code genau wie in versuche/09_training_filter_folds/notebooks/notebook_screening_folds.ipynb ausgeführt (Folds 2 und 3, zweiter Seed). Fold 1 lief mit notebook_screening_fold1.ipynb, dort sind nur die Liste der Läufe und einige Dateinamen anders, Training und Messung sind gleich.
# BeeReal Screening (3-Fold-Vorschlag, 13.11.): Varianten auf einem Fold vergleichen, Wahrheit = automatische Tracks.
# Daten: Ausschnitte aus Stufe B (SAM 3 + ByteTrack-Prinzip), Notebook beereal-ausschnitte.
# Rezept wie bester Stand (Lauf 2/3): DINOv2-small + Linear 128, semi-hard Triplet (Margin 0,2), erst Kopf (1000 Schritte, lr 1e-3),
# dann letzte 4 Blöcke (2000 Schritte, lr 1e-4 / 1e-5), zeitgleiche Pakete, Gelb:Rot 50:50.
# Ausgewertet nach Chan (1 richtige + 10 gleichzeitige Ablenker, >= 600 Bilder Abstand), getrennt für Gelb und Rot.
# Unsicherheit: Bootstrap über Sequenzen (neu) und über Tracks (wie bisher). Gespeichert werden Modelle und Val-Merkmale,
# damit später ohne neues Training gegen die Ground Truth gerechnet werden kann.
# Lauf 1 (14.11.): Fold 1 mit allen vier Varianten. Lauf 2: Folds 2 und 3 für M_ohne, M_streng, G_ohne
# und ein zweiter Seed für M_ohne / M_streng auf allen Folds (zeigt, wie stark das Training allein schwankt).
VARIANTEN = {
    "M_ohne":   dict(hives=["gelb", "rot"], filter="ohne"),
    "M_weich":  dict(hives=["gelb", "rot"], filter="weich"),
    "M_streng": dict(hives=["gelb", "rot"], filter="streng"),
    "G_ohne":   dict(hives=["gelb"], filter="ohne"),
}
LAEUFE = [(f, v, 0) for f in (2, 3) for v in ("M_ohne", "M_streng", "G_ohne")] + [(f, v, 1) for f in (1, 2, 3) for v in ("M_ohne", "M_streng")]
STEPS_HEAD, STEPS_FT, EVAL_EVERY, DRAWS, GAP_FRAMES = 1000, 2000, 500, 20, 600
# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers"], check=False)
# %%
WORKER = r'''
import copy, gc, glob, io, json, os, pickle, random, sys, time
import numpy as np, torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from transformers import AutoModel
CFG = json.loads(sys.argv[1]); OUT = CFG["out"]; START = time.time()
BLOCKS = {"G1": [(2123, 15, 33)], "G2": [(2123, 35, 53)], "G3": [(2123, 55, 58), (2123, 66, 80)],
          "R1": [(2125, 0, 3)], "R2": [(2125, 5, 8)], "R3": [(2125, 10, 13)]}
def seqs(names):
    return {f"rec{r}_seq{i:03d}" for n in names for r, a, b in BLOCKS[n] for i in range(a, b + 1)}
VAL = seqs([f"G{CFG['fold']}", f"R{CFG['fold']}"])
TRAIN = seqs([f"{h}{k}" for h in "GR" for k in (1, 2, 3) if k != CFG["fold"]])
HIVE = {"2123": "gelb", "2125": "rot"}
def log(*a):
    print(f"[{(time.time() - START) / 60:6.1f} min] {CFG['name']}:", *a, flush=True)

JPEG, SEQS = [], {}
for p in sorted(glob.glob("/kaggle/input/**/ausschnitte/*.pkl", recursive=True)):
    sid = os.path.basename(p)[:-4]
    if sid not in VAL | TRAIN:
        continue
    d = pickle.load(open(p, "rb")); trs = {}
    split = "val" if sid in VAL else "train"
    groessen = [c["groesse"] for t in d["tracks"].values() for c in t["crops"]]; med = float(np.median(groessen)) if groessen else 1.0
    for tid, t in d["tracks"].items():
        ids = []
        for c in t["crops"]:
            if split == "train":
                f = CFG["filter"]
                if f == "weich" and not (c["score"] >= 0.5 and c["ueberlappung"] <= 0.2):
                    continue
                if f == "streng" and not (c["score"] >= 0.5 and c["ueberlappung"] <= 0.05 and not c["rand"] and 0.4 <= c["groesse"] / med <= 1.8):
                    continue
            JPEG.append(c["jpg"]); ids.append((c["f"], len(JPEG) - 1))
        if len(ids) >= (4 if split == "train" else 10):
            trs[tid] = dict(crops=ids, moved=t["moved"], first=t["first"], last=t["last"])
    SEQS[sid] = dict(hive=HIVE[sid[3:7]], split=split, tracks=trs)
DATA = {f"{h}_{s}": dict(seqs=sum(v["hive"] == h and v["split"] == s for v in SEQS.values()),
        tracks=sum(len(v["tracks"]) for v in SEQS.values() if v["hive"] == h and v["split"] == s),
        crops=sum(len(t["crops"]) for v in SEQS.values() if v["hive"] == h and v["split"] == s for t in v["tracks"].values()))
        for h in ("gelb", "rot") for s in ("train", "val")}
log("Daten", DATA)

def overlap(a, b):
    return a["first"] <= b["last"] and b["first"] <= a["last"]
def plan_eval(seq_ids, draws, seed=0, max_tracks=None):
    rng = random.Random(seed); trials = []
    for sid in sorted(seq_ids):
        if sid not in SEQS:
            continue
        trs = SEQS[sid]["tracks"]; ids = sorted(trs); rng.shuffle(ids); used = 0
        for tid in ids:
            t = trs[tid]
            if t["last"] - t["first"] < CFG["gap"]:
                continue
            others = [o for o in trs if o != tid and overlap(t, trs[o])]
            if len(others) < 10:
                continue
            qs = [c for c in t["crops"] if c[0] <= t["last"] - CFG["gap"]]
            for _ in range(draws):
                q = rng.choice(qs); pos = [c[1] for c in t["crops"] if c[0] >= q[0] + CFG["gap"]]
                if not pos:
                    continue
                dist = [rng.choice(trs[o]["crops"])[1] for o in rng.sample(others, 10)]
                trials.append(dict(hive=SEQS[sid]["hive"], seq=sid, track=f"{sid}/{tid}", moved=t["moved"], q=q[1], pos=rng.choice(pos), dist=dist))
            used += 1
            if max_tracks and used >= max_tracks:
                break
    return trials
TRIALS = plan_eval(VAL, CFG["draws"]); QUICK = plan_eval(VAL, 3, seed=1, max_tracks=15)
log("Val-Versuche", len(TRIALS), "schnell", len(QUICK))

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
def evaluate(fn, trials, boot=1000, save=None):
    need = sorted({i for t in trials for i in [t["q"], t["pos"], *t["dist"]]})
    E = embed_with(fn, need)
    if save:
        torch.save(dict(idx=need, emb=torch.stack([E[i] for i in need]).half(), trials=trials), save)
    rows = []
    for t in trials:
        q = E[t["q"]]; s_pos = float(q @ E[t["pos"]]); s_d = torch.stack([E[i] for i in t["dist"]]) @ q
        rows.append(dict(hive=t["hive"], seq=t["seq"], track=t["track"], moved=t["moved"], r1=int((s_d >= s_pos).sum()) == 0, rank=1 + int((s_d >= s_pos).sum())))
    res = {}; rng = np.random.default_rng(0)
    for hive in ("gelb", "rot"):
        for name, sub in (("alle", [r for r in rows if r["hive"] == hive]), ("bewegt", [r for r in rows if r["hive"] == hive and r["moved"]])):
            if not sub:
                continue
            r1 = np.array([r["r1"] for r in sub], float)
            per_track = {}; per_seq = {}
            for r in sub:
                per_track.setdefault(r["track"], []).append(r["r1"]); per_seq.setdefault(r["seq"], []).append(r["r1"])
            pt = np.array([np.mean(v) for v in per_track.values()])
            sk = list(per_seq); bs = []
            for _ in range(boot):
                pick = rng.choice(len(sk), len(sk)); vals = np.concatenate([per_seq[sk[i]] for i in pick]); bs.append(vals.mean())
            bt = [rng.choice(pt, len(pt)).mean() for _ in range(boot)]
            res[f"{hive}_{name}"] = dict(versuche=len(sub), tracks=len(pt), sequenzen=len(sk), rank1=round(float(r1.mean()), 3),
                rank3=round(float(np.mean([r["rank"] <= 3 for r in sub])), 3),
                ci_sequenzen=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                ci_tracks=[round(float(np.percentile(bt, 2.5)), 3), round(float(np.percentile(bt, 97.5)), 3)],
                rank1_pro_sequenz={s: round(float(np.mean(v)), 3) for s, v in per_seq.items()})
    return res

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.m = AutoModel.from_pretrained("facebook/dinov2-small"); self.dim = self.m.config.hidden_size
    def forward(self, x):
        return self.m(pixel_values=x).pooler_output
class ReID(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.bb = Net(); self.head = torch.nn.Linear(self.bb.dim, 128)
    def freeze(self, blocks=0):
        for p in self.bb.parameters(): p.requires_grad = False
        if blocks:
            for b in list(self.bb.m.encoder.layer)[-blocks:]:
                for p in b.parameters(): p.requires_grad = True
            for n, p in self.bb.m.named_parameters():
                if n.split(".")[0] in ("norm", "layernorm"):
                    p.requires_grad = True
    def forward(self, x):
        return torch.nn.functional.normalize(self.head(self.bb(x)), dim=1)
def model_fn(m):
    def fn(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return m(x.to("cuda:0")).float()
    return fn
def semihard_triplet(e, y, margin=0.2):
    D = (2 - 2 * e @ e.T).clamp(min=1e-12).sqrt(); same = y[:, None] == y[None]
    a, p = (same & ~torch.eye(len(y), dtype=torch.bool, device=y.device)).nonzero(as_tuple=True)
    dap, dan = D[a, p], D[a]; semi = ~same[a] & (dan > dap[:, None]) & (dan < dap[:, None] + margin); has = semi.any(1)
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
    def __init__(self, hives, steps, P, K, seed):
        self.steps, self.P, self.K, self.rng = steps, P, K, random.Random(seed)
        self.by = {h: [s for s, v in SEQS.items() if v["split"] == "train" and v["hive"] == h and len(v["tracks"]) >= 8] for h in hives}
        self.by = {h: v for h, v in self.by.items() if v}
    def __len__(self): return self.steps
    def __iter__(self):
        for _ in range(self.steps):
            h = self.rng.choice(sorted(self.by)); sid = self.rng.choice(self.by[h]); trs = SEQS[sid]["tracks"]; pool = sorted(trs)
            for _ in range(20):
                t0 = trs[self.rng.choice(pool)]; at = self.rng.randint(t0["first"], t0["last"])
                alive = [i for i in pool if trs[i]["first"] <= at <= trs[i]["last"]]
                if len(alive) >= 8:
                    break
            batch = []
            for tid in self.rng.sample(alive, min(self.P, len(alive))):
                batch += [c[1] for c in self.rng.choices(trs[tid]["crops"], k=self.K)]
            yield batch
def quick_score(m):
    m.eval(); r = evaluate(model_fn(m), QUICK, boot=1); m.train()
    v = [x["rank1"] for k, x in r.items() if k.endswith("_alle")]
    return float(np.mean(v)) if v else 0.0
RES, HIST = {}, {}
def train(m, steps, P, K, blocks, lr_head, lr_bb, name):
    torch.manual_seed(CFG["seed"]); m.to("cuda:0").train(); m.freeze(blocks)
    groups = [{"params": m.head.parameters(), "lr": lr_head}]
    if blocks:
        groups.append({"params": [p for p in m.bb.parameters() if p.requires_grad], "lr": lr_bb})
    opt = torch.optim.AdamW(groups, weight_decay=0.01); scaler = torch.amp.GradScaler("cuda")
    dl = torch.utils.data.DataLoader(TrainSet(), batch_sampler=Sampler(CFG["hives"], steps, P, K, CFG["seed"]), num_workers=3, pin_memory=True)
    best, best_state, hist = -1, None, []
    for step, (x, idx) in enumerate(dl, 1):
        y = torch.arange(len(x)).div(K, rounding_mode="floor").to("cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            e = m(x.to("cuda:0", non_blocking=True))
        loss = semihard_triplet(e.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % CFG["eval_every"] == 0 or step == steps:
            s = quick_score(m); hist.append(dict(step=step, loss=round(float(loss), 4), val_rank1=round(s, 3))); log(name, hist[-1])
            if s > best:
                best, best_state = s, copy.deepcopy({k: v.cpu() for k, v in m.state_dict().items()})
    m.load_state_dict(best_state); m.eval(); HIST[name] = hist
    RES[name] = evaluate(model_fn(m), TRIALS, save=OUT + f"val_merkmale_{name}.pt")
    torch.save({k: v.half() for k, v in m.state_dict().items()}, OUT + f"modell_{name}.pt")
    json.dump(dict(cfg=CFG, daten=DATA, ergebnisse=RES, verlauf=HIST), open(OUT + f"ergebnis_{CFG['name']}.json", "w"), indent=1)
    log(name, "fertig", {k: v["rank1"] for k, v in RES[name].items() if k.endswith("_alle")})
    return m

if CFG.get("nullmodell"):
    bb = Net().to("cuda:0").eval()
    RES["dinov2_untrainiert"] = evaluate(model_fn(bb), TRIALS, save=OUT + f"val_merkmale_dinov2_untrainiert_f{CFG['fold']}.pt")
    RES["pixel_32x32"] = evaluate(lambda x: torch.nn.functional.interpolate(x, size=32, mode="bilinear").flatten(1), TRIALS)
    del bb; gc.collect(); torch.cuda.empty_cache()
m = train(ReID(), CFG["steps_head"], 32, 4, 0, 1e-3, 0, CFG["name"] + "_kopf")
m = train(m, CFG["steps_ft"], 16, 4, 4, 1e-4, 1e-5, CFG["name"] + "_trainiert")
log("ENDE")
'''
# %%
import json, os, subprocess, sys, threading, time
START = time.time(); OUT = "/kaggle/working/"
open(OUT + "worker.py", "w").write(WORKER)
gesehen = set(); jobs = []
for f, n, sd in LAEUFE:   # Nullmodelle (untrainiert, Pixel) einmal pro Fold, im ersten Job dieses Folds (Fold 1 hat sie schon)
    jobs.append(dict(name=f"{n}_f{f}_s{sd}", fold=f, seed=sd, out=OUT, steps_head=STEPS_HEAD, steps_ft=STEPS_FT, eval_every=EVAL_EVERY,
                     draws=DRAWS, gap=GAP_FRAMES, nullmodell=(f != 1 and f not in gesehen), **VARIANTEN[n]))
    gesehen.add(f)
lock = threading.Lock()
def gpu(g):
    while True:
        with lock:
            if not jobs:
                return
            j = jobs.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        p = subprocess.Popen([sys.executable, OUT + "worker.py", json.dumps(j)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for z in p.stdout:
            print(f"GPU {g} |", z.rstrip(), flush=True)
        p.wait()
ths = [threading.Thread(target=gpu, args=(g,)) for g in range(2)]
[t.start() for t in ths]; [t.join() for t in ths]
# Zusammenfassung
import glob, pandas as pd
rows = []
for p in sorted(glob.glob(OUT + "ergebnis_*.json")):
    d = json.load(open(p))
    for modell, r in d["ergebnisse"].items():
        for gruppe, v in r.items():
            rows.append(dict(fold=d["cfg"]["fold"], seed=d["cfg"]["seed"], modell=modell, gruppe=gruppe, rank1=v["rank1"], ci_sequenzen=str(v["ci_sequenzen"]), ci_tracks=str(v["ci_tracks"]),
                             tracks=v["tracks"], sequenzen=v["sequenzen"], versuche=v["versuche"]))
tb = pd.DataFrame(rows); tb.to_csv(OUT + "results.csv", index=False)
print(tb.to_string(index=False), flush=True)
print("FERTIG", round((time.time() - START) / 3600, 2), "h", flush=True)
