"""Gepaarter Vergleich der Trainingsvarianten über die Folds und Siegregel (Versuch 09, Paper 1 Abschnitt 2).

Jede Variante wird gegen die einfachere Variante M_ohne verglichen, und zwar gepaart: gleicher Fold, gleicher Seed,
dieselben Val-Sequenzen. Pro Sequenz wird die Differenz der Rank-1-Werte gebildet. Der 95-%-Bereich kommt aus einem
Bootstrap, der ganze Sequenzen mit Zurücklegen zieht (1000 Wiederholungen).

Siegregel: Eine Variante gilt nur als besser, wenn sie M_ohne in allen drei Folds übertrifft.

Eingabe sind die Dateien ergebnis_<variante>[_f<fold>_s<seed>].json aus den Notebooks beereal-screening (Fold 1)
und beereal-screening-folds (Folds 2 und 3, zweiter Seed). Darin steht rank1_pro_sequenz je Modell und Aufnahme.

Aufruf:  python vergleich_folds.py <ordner mit ergebnis_*.json> [weitere Ordner ...]
"""
import glob, json, os, re, sys
import numpy as np

BASIS = "M_ohne"
BOOT = 1000


def laden(ordner):
    """Liest alle Ergebnisdateien. Schlüssel: (variante, fold, seed). Fold 1 ohne Endung zählt als Fold 1, Seed 0."""
    daten = {}
    for o in ordner:
        for p in glob.glob(os.path.join(o, "ergebnis_*.json")):
            d = json.load(open(p, encoding="utf-8"))
            name = re.sub(r"_f\d+_s\d+$", "", os.path.basename(p)[len("ergebnis_"):-len(".json")])
            daten[(name, d["cfg"]["fold"], d["cfg"]["seed"])] = d["ergebnisse"]
    return daten


def trainiert(ergebnisse):
    """Werte des fertig trainierten Modells (Schlüssel endet auf _trainiert, z. B. M_ohne_f2_s0_trainiert)."""
    return next(v for k, v in ergebnisse.items() if k.endswith("_trainiert"))


def gepaart(a, b, seed=0):
    """Mittlere Differenz a - b über gemeinsame Sequenzen, Bootstrap-Bereich über Sequenzen, Anzahl Sequenzen mit a > b."""
    seqs = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in seqs])
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)]
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((diff > 0).sum()), len(diff)


daten = laden(sys.argv[1:])
varianten = sorted({v for v, _, _ in daten} - {BASIS})
for hive in ("gelb", "rot"):
    print(f"\n=== {hive.upper()} (Rank-1-Differenz zu {BASIS} in Prozentpunkten)")
    for v in varianten:
        siege = {}
        for (name, fold, seed) in sorted(k for k in daten if k[0] == v):
            if (BASIS, fold, seed) not in daten:
                continue
            a = trainiert(daten[(v, fold, seed)])[f"{hive}_alle"]["rank1_pro_sequenz"]
            b = trainiert(daten[(BASIS, fold, seed)])[f"{hive}_alle"]["rank1_pro_sequenz"]
            m, lo, hi, besser, n = gepaart(a, b)
            print(f"{v:10s} Fold {fold} Seed {seed}: {100 * m:+.1f} [{100 * lo:+.1f} bis {100 * hi:+.1f}], "
                  f"besser in {besser} von {n} Sequenzen")
            siege.setdefault(fold, []).append(m > 0)
        laeufe = [s for f in siege.values() for s in f]
        alle_folds = all(any(f) for f in siege.values()) and len(siege) == 3
        print(f"{v:10s} besser in {sum(laeufe)} von {len(laeufe)} Läufen; "
              f"Siegregel (alle drei Folds): {'erfüllt' if alle_folds and all(laeufe) else 'nicht erfüllt'}")
