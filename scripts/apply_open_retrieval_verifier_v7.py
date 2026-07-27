#!/usr/bin/env python3
"""Train-only calibrated, all-OPEN same-image QA retrieval verifier.

The gate is calibrated by leave-one-question-out retrieval on TRAIN. Test
answers are carried through for later scoring only and never enter retrieval,
threshold selection, or answer selection.
"""
import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def norm(x):
    return " ".join(re.findall(r"[a-z0-9]+", str(x).lower()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--test",type=Path,required=True)
    ap.add_argument("--baseline",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--calibration",type=Path,required=True)
    ap.add_argument("--min-train-precision",type=float,default=0.90)
    a=ap.parse_args()
    train=json.loads(a.train.read_text()); test=json.loads(a.test.read_text())
    baseline=json.loads(a.baseline.read_text())
    open_train=[r for r in train if norm(r["answer"]) not in {"yes","no"}]
    by_image=defaultdict(list)
    for r in open_train: by_image[r["image_name"]].append(r)
    corpus=[r["question"] for r in open_train]
    word=TfidfVectorizer(ngram_range=(1,2),stop_words="english",sublinear_tf=True).fit(corpus)
    char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,
                         sublinear_tf=True).fit(corpus)

    def retrieve(question,pool):
        texts=[question]+[x["question"] for x in pool]
        ws=cosine_similarity(word.transform(texts[:1]),word.transform(texts[1:]))[0]
        cs=cosine_similarity(char.transform(texts[:1]),char.transform(texts[1:]))[0]
        scores=.6*ws+.4*cs; i=int(scores.argmax())
        return pool[i],float(scores[i])

    loo=[]
    for row in open_train:
        pool=[x for x in by_image[row["image_name"]] if int(x["qid"])!=int(row["qid"])]
        if not pool: continue
        candidate,score=retrieve(row["question"],pool)
        loo.append((score,norm(candidate["answer"])==norm(row["answer"])))
    trials=[]
    for threshold in np.linspace(0.30,0.99,139):
        selected=[ok for score,ok in loo if score>=threshold]
        if selected:
            trials.append({"threshold":float(threshold),"coverage":len(selected),
                           "correct":int(sum(selected)),
                           "precision":float(sum(selected)/len(selected))})
    viable=[x for x in trials if x["precision"]>=a.min_train_precision]
    if not viable: raise RuntimeError("No threshold reaches requested TRAIN precision")
    best=max(viable,key=lambda x:(x["coverage"],x["precision"]))

    test_by_qid={int(r["qid"]):r for r in test}; out=[]; changes=[]
    for base in baseline:
        item=dict(base); row=test_by_qid[int(item["qid"])]
        item["verifier_v7_used"]=False
        if norm(row["answer"]) not in {"yes","no"} and by_image.get(row["image_name"]):
            candidate,score=retrieve(row["question"],by_image[row["image_name"]])
            item.update({"verifier_v7_similarity":score,
                         "verifier_v7_source_qid":int(candidate["qid"]),
                         "verifier_v7_source_question":candidate["question"],
                         "verifier_v7_candidate":candidate["answer"]})
            if score>=best["threshold"] and norm(candidate["answer"])!=norm(item["response"]):
                before=item["response"];item["response"]=candidate["answer"]
                item["verifier_v7_used"]=True
                changes.append({"qid":item["qid"],"question":item["question"],
                                "gold":item["answer"],"before":before,
                                "after":item["response"],"similarity":score,
                                "source_qid":int(candidate["qid"])})
        out.append(item)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    payload={"train_only":True,"test_labels_used_for_selection":False,
             "calibration_method":"TRAIN leave-one-question-out, same image",
             "word_char_weight":[0.6,0.4],"min_train_precision":a.min_train_precision,
             "best":best,"train_queries":len(loo),"test_changes":len(changes)}
    a.calibration.write_text(json.dumps(payload,indent=2)+"\n")
    a.output.with_name(a.output.stem+"_audit.json").write_text(
        json.dumps({"calibration":payload,"changed_items":changes},indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
