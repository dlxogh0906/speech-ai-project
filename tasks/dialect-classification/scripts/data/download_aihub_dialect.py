"""Download approved AI Hub dialect archives, one file at a time.

Credentials come from AIHUB_APIKEY or ~/.aihub_apikey, never command arguments.
AI Hub returns a tar containing zip parts; only regular archive files are kept.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("Repository root not found")
    _root = _root.parent
sys.path.insert(0, str(_root))

import argparse
import json
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(str(tmp), str(path))


def get_key():
    key = os.environ.get("AIHUB_APIKEY", "").strip()
    if not key:
        key = (Path.home() / ".aihub_apikey").read_text().strip()
    if not key:
        raise RuntimeError("AI Hub API key missing")
    return key


def download(datasetkey, file_info, dest, reserve_gib=40):
    filekey = int(file_info.get("fileSn", file_info.get("filekey")))
    expected = int(file_info.get("fileSize", file_info.get("bytes", 0)))
    folder = Path(dest) / str(datasetkey) / str(filekey)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "download.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        archive = folder / old["archive"]
        if archive.is_file() and archive.stat().st_size == old["bytes"]:
            print("[skip] dataset=%s file=%s" % (datasetkey, filekey), flush=True)
            return archive
    # Parts and their merged zip can briefly coexist.
    need = 2 * expected + int(reserve_gib * 2**30)
    if shutil.disk_usage(str(folder)).free < need:
        raise RuntimeError("Insufficient disk space for archive %s (need %.1f GiB free)" % (filekey, need / 2**30))
    key = get_key()
    url = "https://api.aihub.or.kr/down/0.6/%s.do?fileSn=%s" % (datasetkey, filekey)
    request = urllib.request.Request(url, headers={"apikey": key, "User-Agent": "speech-ai-dialect/1.0"})
    print("[download] dataset=%s file=%s size=%.2f GB" % (datasetkey, filekey, expected / 1e9), flush=True)
    started = time.time()
    names = []
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if "tar" not in response.headers.get("Content-Type", ""):
                message = response.read(4096).decode("utf-8", "replace").replace(key, "<redacted>")
                raise RuntimeError("AI Hub returned a non-tar response: " + message)
            with tarfile.open(fileobj=response, mode="r|*") as bundle:
                for member in bundle:
                    if not member.isfile():
                        continue
                    name = Path(member.name.replace("\\", "/")).name
                    if not re.search(r"\.zip(?:\.part\d+)?$", name, re.I):
                        raise RuntimeError("Unexpected archive member: " + name)
                    if name in names:
                        raise RuntimeError("Duplicate archive member: " + name)
                    target = folder / name
                    with bundle.extractfile(member) as source, target.with_name(name + ".tmp").open("wb") as sink:
                        shutil.copyfileobj(source, sink, 4 * 1024**2)
                    os.replace(str(target.with_name(name + ".tmp")), str(target))
                    names.append(name)
                    print("[part] file=%s bytes=%s elapsed=%.0fs" % (filekey, target.stat().st_size, time.time() - started), flush=True)
    except urllib.error.HTTPError as exc:
        message = exc.read(4096).decode("utf-8", "replace").replace(key, "<redacted>")
        raise RuntimeError("AI Hub HTTP %s: %s" % (exc.code, message)) from None
    whole = [folder / n for n in names if n.lower().endswith(".zip")]
    parts = [folder / n for n in names if re.search(r"\.zip\.part\d+$", n, re.I)]
    if whole and not parts and len(whole) == 1:
        archive = whole[0]
    elif parts and not whole:
        prefixes = {re.sub(r"\.part\d+$", "", p.name) for p in parts}
        if len(prefixes) != 1:
            raise RuntimeError("Multiple zip prefixes in one API file")
        archive = folder / next(iter(prefixes))
        parts.sort(key=lambda p: int(re.search(r"\.part(\d+)$", p.name).group(1)))
        with archive.with_name(archive.name + ".tmp").open("wb") as sink:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, sink, 4 * 1024**2)
        os.replace(str(archive.with_name(archive.name + ".tmp")), str(archive))
        for part in parts:
            part.unlink()
    else:
        raise RuntimeError("Expected exactly one zip, possibly split into parts")
    actual = archive.stat().st_size
    if expected and actual != expected:
        raise RuntimeError("Archive size mismatch: expected %s, got %s" % (expected, actual))
    write_json(marker, {"datasetkey": datasetkey, "filekey": filekey, "archive": archive.name, "bytes": actual, "elapsed_seconds": round(time.time() - started, 1)})
    print("[complete] dataset=%s file=%s archive=%s" % (datasetkey, filekey, archive), flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", default="/data/speech-ai/tmp/aihub-dialect-catalog")
    parser.add_argument("--dest", default="/data/speech-ai/corpus/dialect_10pct_20261010/archives")
    parser.add_argument("--mode", choices=["labels", "plan"], default="labels")
    parser.add_argument("--plan", default="")
    parser.add_argument("--reserve-gib", type=float, default=40)
    args = parser.parse_args()
    if args.mode == "labels":
        jobs = []
        for dataset in [118, 119, 120, 121, 122]:
            files = json.loads((Path(args.catalog_dir) / ("%s-files.json" % dataset)).read_text())["list"]
            jobs.extend((dataset, info) for info in files if "라벨링" in info["fileStreCours"])
    else:
        plan = json.loads(Path(args.plan).read_text())
        jobs = [(row["datasetkey"], info) for row in plan["datasets"] for info in row["selected_files"]]
    for dataset, info in jobs:
        download(dataset, info, args.dest, args.reserve_gib)


if __name__ == "__main__":
    main()
