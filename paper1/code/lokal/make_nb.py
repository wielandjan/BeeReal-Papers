# BeeReal Paper 1, Hilfsmittel: baut aus einem Kaggle-Skript ein Notebook.
"""Baut aus einer .py-Datei (Zellen getrennt durch '# %%') ein Kaggle-Notebook-Verzeichnis.
Aufruf: python make_nb.py <script.py> <zielordner> <kernel-id> [dataset ...]"""
import json, sys
from pathlib import Path

src, dst, kid, datasets = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
dst.mkdir(parents=True, exist_ok=True)
cells = [c.strip("\n") for c in src.read_text(encoding="utf-8").split("# %%") if c.strip()]
nb = {
    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
}
(dst / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": kid, "title": kid.split("/")[1], "code_file": "notebook.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": datasets,
    "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
(dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"{len(cells)} Zellen -> {dst}")
