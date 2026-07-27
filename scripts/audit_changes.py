#!/usr/bin/env python3
import argparse,json,re
from collections import Counter
from pathlib import Path
def norm(x):return " ".join(re.findall(r"[a-z0-9]+",str(x).lower()))
ap=argparse.ArgumentParser();ap.add_argument("before",type=Path);ap.add_argument("after",type=Path)
ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
x={int(r["qid"]):r for r in json.loads(a.before.read_text())}
y={int(r["qid"]):r for r in json.loads(a.after.read_text())};counts=Counter();items=[]
for q,b in x.items():
    z=y[q]
    if norm(b["response"])==norm(z["response"]):continue
    bc=norm(b["response"])==norm(b["answer"]);ac=norm(z["response"])==norm(z["answer"])
    kind="fixed" if ac and not bc else "broke" if bc and not ac else "wrong_to_wrong" if not bc else "correct_to_correct"
    counts[kind]+=1;items.append({"qid":q,"question":b["question"],"gold":b["answer"],
        "before":b["response"],"after":z["response"],"outcome":kind})
payload={"changes":len(items),"counts":dict(counts),
         "exact_net":counts["fixed"]-counts["broke"],"items":items}
a.out.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
