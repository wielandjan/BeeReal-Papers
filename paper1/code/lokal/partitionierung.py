"""Datenpartitionierung beider Aufnahmen (Paper 1, Abschnitt 2 und Tab. 1) und Prüfung der Regeln.

Jede Aufnahme ist zeitlich in drei Blöcke und einen Testbereich geteilt. Zwischen zwei Bereichen liegt je eine
ungenutzte Sequenz als Puffer (30 s), damit benachbarte, fast gleiche Bilder nicht in Training und Validierung
bzw. Test landen. Die Blöcke sind dieselben wie BLOCKS in code/kaggle/beereal_screening.py.

Fold k validiert auf Block k beider Aufnahmen und trainiert gemeinsam auf den beiden anderen Blöcken.
Der Testbereich wird erst ganz am Ende ausgewertet.

Aufruf:  python partitionierung.py      (gibt die Tabelle aus und bricht ab, wenn eine Regel verletzt ist)
"""

# Sequenznummern je Bereich, jeweils (erste, letzte) einschließlich
BEREICHE = {
    "gelb": {   # Aufnahme 2123, im Datensatz die Sequenzen 015-089 (75 Stück)
        "Block 1": [(15, 33)],
        "Block 2": [(35, 53)],
        "Block 3": [(55, 58), (66, 80)],
        "Test":    [(60, 64), (82, 89)],
        "Puffer":  [(34, 34), (54, 54), (59, 59), (65, 65), (81, 81)],
    },
    "rot": {    # Aufnahme 2125, Sequenzen 000-020 (21 Stück); in 000 wird bis etwa 15 s die Kamera aufgebaut
        "Block 1": [(0, 3)],
        "Block 2": [(5, 8)],
        "Block 3": [(10, 13)],
        "Test":    [(15, 20)],
        "Puffer":  [(4, 4), (9, 9), (14, 14)],
    },
}


def nummern(spannen):
    return [i for a, b in spannen for i in range(a, b + 1)]


def fold(k):
    """Sequenzen für Training und Validierung in Fold k (1, 2 oder 3)."""
    val = {h: nummern(b[f"Block {k}"]) for h, b in BEREICHE.items()}
    train = {h: [i for j in (1, 2, 3) if j != k for i in nummern(b[f"Block {j}"])] for h, b in BEREICHE.items()}
    return train, val


def pruefen():
    for hive, bereiche in BEREICHE.items():
        zuordnung = {}
        for name, spannen in bereiche.items():
            for i in nummern(spannen):
                assert i not in zuordnung, f"{hive} {i:03d} doppelt ({zuordnung.get(i)} und {name})"
                zuordnung[i] = name
        # Zwei verschiedene genutzte Bereiche dürfen nie direkt aneinandergrenzen, dazwischen liegt ein Puffer
        folge = sorted(zuordnung)
        for a, b in zip(folge, folge[1:]):
            if b == a + 1 and "Puffer" not in (zuordnung[a], zuordnung[b]):
                assert zuordnung[a] == zuordnung[b], f"{hive}: {a:03d} ({zuordnung[a]}) grenzt an {b:03d} ({zuordnung[b]})"
    # Folds: Validierung und Training überschneiden sich nie
    for k in (1, 2, 3):
        train, val = fold(k)
        for h in BEREICHE:
            assert not set(train[h]) & set(val[h]), f"Fold {k} {h}: Überschneidung"


if __name__ == "__main__":
    pruefen()
    for hive, bereiche in BEREICHE.items():
        print(hive.upper())
        for name, spannen in bereiche.items():
            print(f"  {name:8s} {len(nummern(spannen)):3d} Sequenzen  " + ", ".join(
                f"{a:03d}" if a == b else f"{a:03d}-{b:03d}" for a, b in spannen))
    for k in (1, 2, 3):
        train, val = fold(k)
        print(f"Fold {k}: Val Gelb {len(val['gelb'])} + Rot {len(val['rot'])}, "
              f"Training Gelb {len(train['gelb'])} + Rot {len(train['rot'])} Sequenzen")
    print("Alle Regeln erfüllt.")
