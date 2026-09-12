#!/usr/bin/env python3
"""Fetch the pinned official UMI WBC code/assets and released reference data.

Existing files are preserved. Hardware submodules and documentation media are
not needed for the WBC runtime and are not downloaded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import shutil
from urllib.request import Request, urlopen
import zipfile

SHA = "d75c9c182d8044dadf53043612da2ffbf1936a97"
REPO = "real-stanford/umi-on-legs"


def download(url, target):
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    request = Request(url, headers={"User-Agent": "PawCerto-reference-fetch"})
    with urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(target)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tree-json", type=Path, help="Reuse an existing GitHub tree response")
    args = parser.parse_args()
    root = args.root.resolve()
    tree_path = args.tree_json or root / "downloads/umi-source-tree.json"
    download(f"https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1", tree_path)
    tree = json.loads(tree_path.read_text())
    if tree["sha"] != SHA or tree.get("truncated"):
        raise ValueError("Expected the complete pinned UMI source tree")
    files = [entry["path"] for entry in tree["tree"] if entry["type"] == "blob"
             and (entry["path"].startswith("mani-centric-wbc/")
                  or entry["path"] in ("README.md", "LICENSE"))
             and not entry["path"].endswith((".gif", ".mp4"))]
    def fetch_source(path):
        return download(f"https://raw.githubusercontent.com/{REPO}/{SHA}/{path}",
                        root / "third_party/umi-on-legs" / path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        added = sum(pool.map(fetch_source, files))
    print(f"Source {SHA}: {added} downloaded, {len(files) - added} existing files preserved")
    for name in ("data", "checkpoints"):
        archive = root / f"downloads/umi-{name}.zip"
        download(f"https://real.stanford.edu/umi-on-legs/wbc/{name}.zip", archive)
        with zipfile.ZipFile(archive) as bundle:
            bad = bundle.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"CRC failed: {bad}")
            extracted = 0
            for member in bundle.infolist():
                target = root / "reference" / member.filename
                if (root / "reference").resolve() not in target.resolve().parents:
                    raise ValueError(f"Archive path outside reference directory: {member.filename}")
                if not target.exists():
                    bundle.extract(member, root / "reference")
                    extracted += 1
        print(f"{name}: ZIP CRC passed; {extracted} missing entries extracted")
    converted = 0
    for config_path in sorted((root / "reference/checkpoints").rglob("config.pkl")):
        json_path = config_path.with_suffix(".json")
        if json_path.exists():
            continue
        # These configs come from the official checkpoint archive above.
        with config_path.open("rb") as stream:
            config = pickle.load(stream)
        content = json.dumps(config, indent=2, allow_nan=False) + "\n"
        json_path.write_text(content)
        converted += 1
    print(f"Checkpoint configs: {converted} JSON files generated; existing JSON preserved")


if __name__ == "__main__":
    main()
