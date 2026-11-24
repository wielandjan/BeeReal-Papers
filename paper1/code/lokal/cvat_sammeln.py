# BeeReal Paper 1, Versuch 10.
"""Fasst mehrere CVAT-Pakete (gleich viele Bilder je Paket) zu einem Paket zusammen: eine Aufgabe, ein Job pro Ausschnitt.
Grund: Der kostenlose CVAT-Plan erlaubt nur 3 Aufgaben pro Organisation.

Aufruf: python cvat_sammeln.py <paketordner> <name_neu> <ausgabeordner>
Bilder heißen danach <ausschnitt>_<nr>.jpg (sortiert = Reihenfolge), Track-IDs werden je Ausschnitt verschoben.
In CVAT mit segment_size = Bilder pro Ausschnitt hochladen (cvat_hochladen.py macht das, wenn <name>_info.json 'segment_size' hat).
"""
import glob, json, os, sys, zipfile

ordner, name, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
pakete = sorted(p[: -len("_bilder.zip")] for p in glob.glob(os.path.join(ordner, "*_bilder.zip")))
zb = zipfile.ZipFile(os.path.join(out, f"{name}_bilder.zip"), "w"); zeilen = []; versatz_bild = versatz_id = 0; teile = []
for p in pakete:
    kurz = os.path.basename(p); zi = zipfile.ZipFile(p + "_bilder.zip"); bilder = sorted(zi.namelist())
    for k, b in enumerate(bilder):
        zb.writestr(f"{kurz}_{k + 1:06d}.jpg", zi.read(b))
    max_id = 0
    if os.path.exists(p + "_vorschlaege_mot.zip"):
        for z in zipfile.ZipFile(p + "_vorschlaege_mot.zip").read("gt/gt.txt").decode().split():
            f, i, *rest = z.split(","); max_id = max(max_id, int(i))
            zeilen.append(",".join([str(int(f) + versatz_bild), str(int(i) + versatz_id)] + rest))
    teile.append(dict(ausschnitt=kurz, bilder=len(bilder), erstes_bild=versatz_bild + 1))
    versatz_bild += len(bilder); versatz_id += max_id
zb.close()
if zeilen:
    with zipfile.ZipFile(os.path.join(out, f"{name}_vorschlaege_mot.zip"), "w") as za:
        za.writestr("gt/gt.txt", "\n".join(zeilen) + "\n"); za.writestr("gt/labels.txt", "bee\n")
groessen = {t["bilder"] for t in teile}
json.dump(dict(ausschnitte=teile, segment_size=teile[0]["bilder"] if len(groessen) == 1 else None), open(os.path.join(out, f"{name}_info.json"), "w"), indent=1)
print(name, len(teile), "Ausschnitte,", versatz_bild, "Bilder,", len(zeilen), "Kästen, Jobs à", teile[0]["bilder"])
