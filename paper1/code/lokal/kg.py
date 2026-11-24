# BeeReal Paper 1, Hilfsmittel: Kaggle-Befehlszeile hinter einem Netz mit eigener Zertifizierungsstelle.
"""Kaggle CLI mit Windows-Zertifikatsspeicher (Netzwerk mit eigener Zertifizierungsstelle)."""
import sys
import truststore

truststore.inject_into_ssl()
from kaggle.cli import main

sys.argv = ["kaggle"] + sys.argv[1:]
main()
