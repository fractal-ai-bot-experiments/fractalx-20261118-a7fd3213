"""Steps 2-4: hidden state at the word's last token, every layer, under 12 carrier sentences in
3 disjoint sets; per carrier and layer centre over all words, then average over the set.
Also the model's input embeddings (mean over the word's tokens with a leading space).
Uses recipes.interp (TransformerLens bridge, hook blocks.{L}.hook_resid_post)."""
import argparse, json
import numpy as np, torch
from recipes.interp import load_bridge, span_residuals

ANCHORS = {"hand_arm": ["hand", "arm"], "foot_leg": ["foot", "leg"], "mouth": ["mouth", "throat"]}
PARTS = {"hand_arm": ["finger", "thumb", "wrist", "fist", "knuckle", "elbow", "fingernail", "bicep"],
         "foot_leg": ["toe", "heel", "ankle", "knee", "shin", "thigh", "toenail", "kneecap"],
         "mouth": ["lip", "tongue", "teeth", "jaw", "gums", "tonsil", "tooth", "palate"]}
# Neutral carriers ending in the word; no action verb, no body part. Three disjoint sets (seeds 0-2).
CARRIERS = [
    ["The next word on the list is", "Here is the word", "The answer to the crossword clue is",
     "The label on the box says"],
    ["Item number seven:", "The last entry in the catalogue was", "Today's vocabulary word is",
     "The topic of the next page is"],
    ["The title of the file is", "Next on the inventory:", "The key term in this section is",
     "In the picture:"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    nouns = [json.loads(l) for l in open("/scratch/data/nouns.jsonl")]
    words = [n["word"] for n in nouns] + [w for v in ANCHORS.values() for w in v] + [w for v in PARTS.values() for w in v]
    assert len(set(words)) == len(words), "duplicate word"
    bridge, model, tok = load_bridge(a.model)
    L = int(bridge.cfg.n_layers)
    layers = list(range(L))
    sets = []
    for cset in CARRIERS:
        acc = None
        for carrier in cset:
            texts = [" " + w for w in words]
            prefixes = [carrier] * len(words)
            outs = []
            for stack, span in span_residuals(bridge, tok, texts, layers, prefixes=prefixes, batch_size=a.batch):
                last = span.long().cumsum(dim=1).argmax(dim=1)
                outs.append(stack[torch.arange(len(last)), :, last])  # [b, L, d]
            h = torch.cat(outs)  # [W, L, d]
            h = h - h.mean(dim=0, keepdim=True)  # centre over all words, per carrier and layer
            acc = h if acc is None else acc + h
        sets.append((acc / len(cset)).numpy().astype(np.float32))
    H = np.stack(sets)  # [3, W, L, d]
    emb = model.get_input_embeddings().weight
    E = []
    for w in words:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        E.append(emb[ids].float().mean(0).detach().cpu().numpy())
    E = np.stack(E)
    ntok = [len(tok(" " + w, add_special_tokens=False)["input_ids"]) for w in words]
    np.savez(a.out, H=H, E=E, words=np.array(words), ntok=np.array(ntok))
    print("saved", a.out, H.shape, E.shape, "mean tokens/word", np.mean(ntok))


if __name__ == "__main__":
    main()
