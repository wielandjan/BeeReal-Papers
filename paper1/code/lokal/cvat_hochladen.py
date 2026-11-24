# BeeReal Paper 1, Versuch 10. Der Zugang liegt nur lokal beim Nutzer, nie im Repo.
"""Legt für jedes Labeling-Paket eine Aufgabe im CVAT-Projekt an: Bilder hochladen, Vorschläge (MOT 1.1) importieren.

Zugang: Datei %USERPROFILE%\\.cvat_zugang mit einer Zeile (persönlicher Zugangsschlüssel aus CVAT, empfohlen)
oder zwei Zeilen (Benutzername, Passwort). Die Datei legt Jan selbst an, sie wird nie ausgegeben oder ins Repo gelegt.

Aufruf: python cvat_hochladen.py <paketordner> <Train|Validation|Test> [projekt="BeeReal GT"] [nur_paket_name]
Ein Paket = <name>_bilder.zip + <name>_vorschlaege_mot.zip (gebaut mit cvat_paket.py). Vorhandene Aufgaben gleichen Namens werden übersprungen.
"""
import glob, json, os, sys
import truststore; truststore.inject_into_ssl()
from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType

ordner, subset = sys.argv[1], sys.argv[2]
projekt_name = sys.argv[3] if len(sys.argv) > 3 else "BeeReal GT"
nur = sys.argv[4] if len(sys.argv) > 4 else None
zeilen = [z.strip() for z in open(os.path.expanduser("~/.cvat_zugang"), encoding="utf-8").read().splitlines() if z.strip()]
zugang = dict(access_token=zeilen[0]) if len(zeilen) == 1 else dict(credentials=(zeilen[0], zeilen[1]))

with make_client(host="https://app.cvat.ai", **zugang) as client:
    client.organization_slug = os.environ.get("CVAT_ORG", "BeeReal")  # Projekt liegt in der Organisation BeeReal
    projekte = [p for p in client.projects.list() if p.name == projekt_name]
    if not projekte:
        sys.exit(f"Projekt '{projekt_name}' nicht gefunden")
    pid = projekte[0].id
    vorhanden = {t.name for t in client.tasks.list() if t.project_id == pid}
    for bilder in sorted(glob.glob(os.path.join(ordner, "*_bilder.zip"))):
        name = os.path.basename(bilder)[: -len("_bilder.zip")]
        if nur and name != nur:
            continue
        if name in vorhanden:
            print(name, "gibt es schon, übersprungen"); continue
        mot = os.path.join(ordner, f"{name}_vorschlaege_mot.zip")
        info = os.path.join(ordner, f"{name}_info.json")
        seg = json.load(open(info)).get("segment_size") if os.path.exists(info) else None  # ein Job pro Ausschnitt
        task = client.tasks.create_from_data(
            spec={"name": name, "project_id": pid, "subset": subset, **({"segment_size": seg} if seg else {})},
            resource_type=ResourceType.LOCAL, resources=[bilder],
            data_params={"image_quality": 95, "sorting_method": "lexicographical"},
            annotation_path=mot if os.path.exists(mot) else "", annotation_format="MOT 1.1")
        print(name, "angelegt: Aufgabe", task.id, f"https://app.cvat.ai/tasks/{task.id}")
