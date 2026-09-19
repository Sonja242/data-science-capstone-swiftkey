"""Download the exact public model releases used in this comparison."""
import argparse
import json
from neural_predictor import ROOT, REPOS
from huggingface_hub import snapshot_download

REVISIONS = {
    "0.6B": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "1.7B": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
}

parser = argparse.ArgumentParser()
parser.add_argument("--size", choices=REPOS, required=True)
args = parser.parse_args()
repo = REPOS[args.size]
revision = REVISIONS[args.size]
folder = ROOT / "models/neural" / repo.split("/")[1]
snapshot_download(repo_id=repo, revision=revision, token=False, local_dir=folder,
    allow_patterns=["*.json", "*.safetensors", "merges.txt", "*.md"], max_workers=2)
(folder / "download_manifest.json").write_text(json.dumps({
    "repo": repo, "revision": revision, "license": "Apache-2.0"}, indent=2), encoding="utf-8")
print("Downloaded pinned release:", repo, revision)
