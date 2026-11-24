# BeeReal Paper 1, Versuch 09, Vorstufe. Schneidet die Trainingsausschnitte aus den Tracks von Versuch 08.
# BeeReal: Bildausschnitte aus den Stufe-B-Tracks schneiden (CPU-Notebook, kostet keine GPU-Zeit).
# Eingabe: Datensatz beereal-tracks-v2 (tracks_<sid>.json.gz, ByteTrack-Prinzip auf SAM-3-Kästen).
# Grundfilter wie in allen bisherigen Läufen: Kasten mit IoU >= 0,3 zu einem anderen sicheren Kasten fällt weg; Track braucht >= 10 Kästen.
# Pro Track höchstens 40 Ausschnitte, gleichmäßig über die Zeit. Zu jedem Ausschnitt werden die Maße gespeichert,
# damit der Qualitätsfilter (ohne / weich / streng) erst beim Training gewählt werden kann.
MIN_CROPS, MAX_CROPS, IOU_MAX = 10, 40, 0.3
# %%
import glob, re
from huggingface_hub import login
_f = glob.glob("/kaggle/input/**/hf_token*", recursive=True)
if not _f:
    _f = [p for p in glob.glob("/kaggle/input/**/*", recursive=True) if p.endswith((".txt", ".csv"))]
login(re.search(r"hf_[A-Za-z0-9]+", open(_f[0], encoding="utf-8-sig").read()).group(0))
print("HF_LOGIN OK", flush=True)
# %%
import gzip, json, os, pickle, time
from multiprocessing import Pool
import cv2, numpy as np
from huggingface_hub import hf_hub_download
START = time.time(); OUT = "/kaggle/working/ausschnitte/"; os.makedirs(OUT, exist_ok=True)
DATEIEN = sorted(p for p in glob.glob("/kaggle/input/**/tracks_rec*_seq*.json*", recursive=True))
print(len(DATEIEN), "Track-Dateien", flush=True)

def lade(p):
    return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))

def schneide(p):
    t0 = time.time(); d = lade(p); sid = d["sid"]; x0, y0 = d["roi"][0], d["roi"][1]
    if os.path.exists(OUT + f"{sid}.pkl"):
        return sid, "schon da"
    F = {n: i for i, n in enumerate(d["felder"])}
    plan, brauche = {}, set()
    for t in d["tracks"]:
        k = [x for x in t["kaesten"] if x[F["iou_max"]] < IOU_MAX]
        if len(k) < MIN_CROPS:
            continue
        c = np.array([[(x[1] + x[3]) / 2, (x[2] + x[4]) / 2] for x in k]); s = float(np.median([max(x[3] - x[1], x[4] - x[2]) for x in k]))
        wahl = [k[i] for i in np.unique(np.linspace(0, len(k) - 1, MAX_CROPS).astype(int))]
        plan[t["id"]] = dict(first=k[0][0], last=k[-1][0], moved=bool(np.linalg.norm(c.max(0) - c.min(0)) > s), wahl=wahl)
        brauche |= {x[0] for x in wahl}
    cap = cv2.VideoCapture(hf_hub_download("mrkschtr/thehiveisreal", f"videos/{sid}.mp4", repo_type="dataset"))
    bilder, f, letzte = {}, 0, max(brauche) if brauche else -1
    while f <= letzte:
        if f in brauche:
            ok, im = cap.read()
            if ok:
                bilder[f] = im
        else:
            ok = cap.grab()
        if not ok:
            break
        f += 1
    tracks = {}
    for tid, t in plan.items():
        crops = []
        for x in t["wahl"]:
            im = bilder.get(x[0])
            if im is None:
                continue
            a, b, c2, e = int(x[1] + x0), int(x[2] + y0), int(x[3] + x0), int(x[4] + y0)
            ok, jpg = cv2.imencode(".jpg", im[max(b, 0):e, max(a, 0):c2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                crops.append(dict(f=x[0], jpg=jpg.tobytes(), score=x[F["score"]], flaeche=x[F["flaeche"]], fuellgrad=x[F["fuellgrad"]],
                                  ueberlappung=x[F["ueberlappung"]], rand=x[F["rand"]], groesse=max(x[3] - x[1], x[4] - x[2])))
        if len(crops) >= MIN_CROPS:
            tracks[tid] = dict(first=t["first"], last=t["last"], moved=t["moved"], crops=crops)
    pickle.dump(dict(sid=sid, tracks=tracks), open(OUT + f"{sid}.pkl.tmp", "wb")); os.replace(OUT + f"{sid}.pkl.tmp", OUT + f"{sid}.pkl")
    return sid, f"{len(tracks)} Tracks, {sum(len(v['crops']) for v in tracks.values())} Ausschnitte, {time.time() - t0:.0f} s"

with Pool(4) as pool:
    for sid, info in pool.imap_unordered(schneide, DATEIEN):
        print(f"[{(time.time() - START) / 60:6.1f} min]", sid, info, flush=True)
print("FERTIG", round((time.time() - START) / 60, 1), "min,", len(os.listdir(OUT)), "Dateien", flush=True)
