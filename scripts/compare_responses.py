#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
def norm(x):return " ".join(re.findall(r"[a-z0-9]+",str(x).lower()))
ap=argparse.ArgumentParser();ap.add_argument("actual",type=Path);ap.add_argument("expected",type=Path)
a=ap.parse_args();x=json.loads(a.actual.read_text());y=json.loads(a.expected.read_text())
if len(x)!=len(y):raise SystemExit(f"row-count mismatch: {len(x)} != {len(y)}")
bad=[]
for i,(u,v) in enumerate(zip(x,y)):
    if (int(u["qid"]),norm(u["response"]))!=(int(v["qid"]),norm(v["response"])):
        bad.append({"index":i,"qid":u["qid"],"actual":u["response"],"expected":v["response"]})
print(json.dumps({"response_match":not bad,"rows":len(x),"mismatches":bad[:20]},indent=2))
if bad:raise SystemExit(1)
