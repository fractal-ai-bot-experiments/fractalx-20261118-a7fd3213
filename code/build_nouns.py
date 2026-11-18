"""Step 1: tool-noun list from the Lancaster Sensorimotor Norms, WordNet and GloVe filters, labels, split."""
import json, sys
import numpy as np, pandas as pd
from nltk.corpus import wordnet as wn
import gensim.downloader as api

BODY = ["hand", "arm", "foot", "leg", "mouth", "throat", "head", "finger", "toe", "lip", "knee"]
COLS = {"hand_arm": "Hand_arm.mean", "foot_leg": "Foot_leg.mean", "mouth": "Mouth.mean"}
ART = wn.synset("artifact.n.01")
FOOD = {wn.synset("food.n.01"), wn.synset("food.n.02")}


def under(word):
    art, food = False, False
    for s in wn.synsets(word, pos=wn.NOUN):
        hyp = set(s.closure(lambda x: x.hypernyms() + x.instance_hypernyms())) | {s}
        art |= ART in hyp
        food |= bool(hyp & FOOD)
    return art, food


def main():
    df = pd.read_csv("/scratch/data/lancaster.csv")
    df["w"] = df["Word"].astype(str).str.lower()
    df = df[df["w"].str.fullmatch(r"[a-z]+")]
    glove = api.load("glove-wiki-gigaword-300")
    keep = []
    for _, r in df.iterrows():
        w = r["w"]
        if any(b in w for b in BODY) or w not in glove.key_to_index:
            continue
        art, food = under(w)
        if art and not food:
            keep.append(r)
    cand = pd.DataFrame(keep)
    print("candidate artifact nouns:", len(cand))
    out = None
    for margin in (1.0, 0.5):
        rows = []
        for _, r in cand.iterrows():
            vals = {k: float(r[c]) for k, c in COLS.items()}
            top = max(vals, key=vals.get)
            others = [v for k, v in vals.items() if k != top]
            if vals[top] >= 2.5 and vals[top] - max(others) >= margin:
                rows.append(dict(word=r["w"], label=top, **vals))
        lab = pd.DataFrame(rows)
        counts = lab["label"].value_counts().to_dict()
        print("margin", margin, counts)
        n = min(60, min(counts.get(k, 0) for k in COLS))
        if n >= 30 or margin == 0.5:
            out, used_margin = lab, margin
            break
    rng = np.random.default_rng(0)
    chosen = []
    for k in COLS:
        sub = out[out.label == k].sort_values("word")
        idx = rng.permutation(len(sub))[:n]
        part = sub.iloc[idx].copy()
        half = np.zeros(n, dtype=int)
        half[rng.permutation(n)[: n // 2]] = 1  # 1 = selection half
        part["half"] = np.where(half == 1, "selection", "report")
        chosen.append(part)
    res = pd.concat(chosen).sort_values(["label", "word"])
    res.to_json("/scratch/data/nouns.jsonl", orient="records", lines=True)
    meta = dict(n_per_class=int(n), margin=used_margin, n_candidates=int(len(cand)),
                class_counts_before_balance={k: int(v) for k, v in out["label"].value_counts().items()})
    json.dump(meta, open("/scratch/data/nouns_meta.json", "w"), indent=1)
    print(meta)
    for k in COLS:
        print(k, list(res[res.label == k].word))


if __name__ == "__main__":
    main()
