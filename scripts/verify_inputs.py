#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

TRAIN_SHA = "3b0567673ae224b6489ca1bc4ce05cda642223d5cf4f17db8fbf7010c872e00f"
TEST_SHA = "3d32a047592cb680e18054030b38b1a44d6b17105fc62d3088c8655d56a00ced"
BASE_CONFIG_SHA = "c73e1f81f374d903872f2833f066b8cd081037d72f8c0f436d06fa7b7cc82ea5"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--base",type=Path);ap.add_argument("--require-images",action="store_true")
    a=ap.parse_args(); errors=[]
    for name,want in (("trainset.json",TRAIN_SHA),("testset.json",TEST_SHA)):
        p=a.data/name
        if not p.is_file():errors.append(f"missing {p}")
        elif sha(p)!=want:errors.append(f"hash mismatch {p}")
    images=a.data/"images"
    if a.require_images and (not images.is_dir() or len(list(images.iterdir()))!=315):
        errors.append(f"expected exactly 315 image files in {images}")
    if a.base:
        p=a.base/"config.json"
        if not p.is_file():errors.append(f"missing {p}")
        elif sha(p)!=BASE_CONFIG_SHA:errors.append(f"base config hash mismatch {p}")
    payload={"ok":not errors,"data":str(a.data),"base":str(a.base) if a.base else None,
             "errors":errors}
    print(json.dumps(payload,indent=2))
    if errors:raise SystemExit(1)
if __name__=="__main__":main()
