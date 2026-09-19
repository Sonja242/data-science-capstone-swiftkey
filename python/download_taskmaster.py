"""Retrieve the pinned, attributed Taskmaster-1 source for local analysis."""
from pathlib import Path
import hashlib
import json
import requests
from neural_predictor import ROOT

REVISION="d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
EXPECTED="1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d"
folder=ROOT/"data/external/taskmaster"
folder.mkdir(parents=True,exist_ok=True)
base=f"https://raw.githubusercontent.com/google-research-datasets/Taskmaster/{REVISION}/TM-1-2019/"
for source,name in (("self-dialogs.json","self-dialogs.json"),("train-dev-test/train.csv","train.csv"),("README.md","README.md")):
    destination=folder/name
    if not destination.exists():
        temporary=destination.with_suffix(destination.suffix+".part")
        with requests.get(base+source,stream=True,timeout=(15,60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024*1024): output.write(chunk)
        if name=="self-dialogs.json":
            assert hashlib.sha256(temporary.read_bytes()).hexdigest()==EXPECTED
        temporary.replace(destination)
    if name=="self-dialogs.json":
        assert hashlib.sha256(destination.read_bytes()).hexdigest()==EXPECTED
manifest={"revision":REVISION,"repository":"google-research-datasets/Taskmaster",
          "sha256":EXPECTED,"license":"CC BY 4.0",
          "subset":"TM-1-2019 self-dialogs, official training conversation IDs"}
(folder/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print("Pinned Taskmaster source verified. Google LLC, CC BY 4.0; see the downloaded README.")
