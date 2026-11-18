"""Steps 5-9: nearest-body-part readout per layer and seed, layer selection on the selection half,
baselines (input embeddings, GloVe), shuffles, positive control, bootstrap, partial correlation."""
import json
import numpy as np
import gensim.downloader as api
from scipy.stats import rankdata
from recipes.seeds import write_results
from extract import ANCHORS, PARTS

CLASSES = ["hand_arm", "foot_leg", "mouth"]
SEEDS = [0, 1, 2]
TIE = 0.02
rng = np.random.default_rng(0)

nouns = [json.loads(l) for l in open("/scratch/data/nouns.jsonl")]
d = np.load("/scratch/data/acts_8b.npz")
H, E, words = d["H"], d["E"], list(d["words"])
idx = {w: i for i, w in enumerate(words)}
N = len(nouns)
y = np.array([CLASSES.index(n["label"]) for n in nouns])
sel = np.array([n["half"] == "selection" for n in nouns])
rep = ~sel
pc_words = [w for c in CLASSES for w in PARTS[c]]
pc_y = np.array([CLASSES.index(c) for c in CLASSES for _ in PARTS[c]])
noun_ix = [idx[n["word"]] for n in nouns]
pc_ix = [idx[w] for w in pc_words]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cos_to_classes(V):
    """V: [W, d] centred vectors for all words. Returns cosine of every word to each class vector."""
    C = np.stack([V[[idx[a] for a in ANCHORS[c]]].mean(0) for c in CLASSES])
    return unit(V) @ unit(C).T  # [W, 3]


# static baselines, same centring (over all words) and readout
glove = api.load("glove-wiki-gigaword-300")
G = np.stack([glove[w] for w in words]).astype(np.float64)
G -= G.mean(0)
Ec = E.astype(np.float64) - E.mean(0)
cos_glove, cos_emb = cos_to_classes(G), cos_to_classes(Ec)
pred_glove, pred_emb = cos_glove.argmax(1), cos_emb.argmax(1)


def acc(pred, ix, labels, mask=None):
    p = pred[ix]
    ok = p == labels
    return float(ok[mask].mean()) if mask is not None else float(ok.mean())


acc_emb_rep = acc(pred_emb, noun_ix, y, rep)
acc_glove_rep = acc(pred_glove, noun_ix, y, rep)
best_name = "glove_vectors" if acc_glove_rep >= acc_emb_rep else "model_input_embeddings"
pred_best = pred_glove if best_name == "glove_vectors" else pred_emb
acc_best_rep = max(acc_emb_rep, acc_glove_rep)

# hidden-state readout for every seed and layer
S, W, L, D = H.shape
cos_h = np.zeros((S, L, W, 3))
for s in range(S):
    for l in range(L):
        cos_h[s, l] = cos_to_classes(H[s, :, l].astype(np.float64))
pred_h = cos_h.argmax(-1)  # [S, L, W]
acc_sel = np.array([[acc(pred_h[s, l], noun_ix, y, sel) for l in range(L)] for s in range(S)])
acc_rep = np.array([[acc(pred_h[s, l], noun_ix, y, rep) for l in range(L)] for s in range(S)])
acc_pc_layers = np.array([[acc(pred_h[s, l], pc_ix, pc_y) for l in range(L)] for s in range(S)])
crit = acc_sel.mean(0)
best = crit.max()
chosen = int(min(l for l in range(L) if crit[l] >= best - TIE))
print("selection criterion per layer:", np.round(crit, 3).tolist())
print("chosen layer", chosen, "criterion", crit[chosen])

per_seed, pc, shuf, pvals, wrong_acc, partial = {}, {}, {}, {}, {}, {}
yr = y[rep]
rep_ix = np.array(noun_ix)[rep]
best_wrong = pred_best[rep_ix] != yr
for s in SEEDS:
    p = pred_h[s, chosen][rep_ix]
    a = float((p == yr).mean())
    per_seed[str(s)] = a
    pc[str(s)] = acc(pred_h[s, chosen], pc_ix, pc_y)
    null = np.array([(p == rng.permutation(yr)).mean() for _ in range(1000)])
    shuf[str(s)] = float(null.mean())
    pvals[str(s)] = float((1 + (null >= a).sum()) / 1001)
    wrong_acc[str(s)] = float((p[best_wrong] == yr[best_wrong]).mean())
    # Spearman partial correlation, per class: hidden cosine vs Lancaster rating, controlling GloVe cosine
    rs = []
    for k, c in enumerate(CLASSES):
        x = rankdata(cos_h[s, chosen][rep_ix, k])
        r = rankdata([n[c] for n, m in zip(nouns, rep) if m])
        z = rankdata(cos_glove[rep_ix, k])
        Z = np.c_[np.ones_like(z), z]
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
        rs.append(float(np.corrcoef(rx, ry)[0, 1]))
        partial[f"partial_spearman_{c}_seed{s}"] = rs[-1]
    partial[f"partial_spearman_mean_seed{s}"] = float(np.mean(rs))

# paired bootstrap over reporting nouns: seed-mean accuracy minus stronger baseline
ok_h = np.stack([pred_h[s, chosen][rep_ix] == yr for s in SEEDS]).mean(0)
ok_b = (pred_best[rep_ix] == yr).astype(float)
nr = len(yr)
boots = np.array([(ok_h[b] - ok_b[b]).mean() for b in (rng.integers(0, nr, nr) for _ in range(10000))])
lo, hi = np.percentile(boots, [2.5, 97.5])

other = {
    "chosen_layer": chosen,
    "accuracy_minus_best_baseline_mean": float(np.mean(list(per_seed.values())) - acc_best_rep),
    "bootstrap_diff_ci_low": float(lo), "bootstrap_diff_ci_high": float(hi),
    "glove_accuracy_on_body_part_parts": acc(pred_glove, pc_ix, pc_y),
    "embedding_accuracy_on_body_part_parts": acc(pred_emb, pc_ix, pc_y),
    "embedding_accuracy_selection_half": acc(pred_emb, noun_ix, y, sel),
    "glove_accuracy_selection_half": acc(pred_glove, noun_ix, y, sel),
    "best_baseline_accuracy_on_its_wrong_nouns_count": int(best_wrong.sum()),
    "best_layer_reporting_half_mean_accuracy_any_layer": float(acc_rep.mean(0).max()),
}
for s in SEEDS:
    other[f"shuffle_p_value_seed{s}"] = pvals[str(s)]
    other[f"accuracy_on_baseline_wrong_nouns_seed{s}"] = wrong_acc[str(s)]
other.update(partial)
# per-class accuracy and confusion matrix (seed-pooled counts) at the chosen layer
for k, c in enumerate(CLASSES):
    m = yr == k
    other[f"per_class_accuracy_{c}"] = float(np.mean([(pred_h[s, chosen][rep_ix][m] == k).mean() for s in SEEDS]))
    other[f"per_class_accuracy_{c}_best_baseline"] = float((pred_best[rep_ix][m] == k).mean())
    for j, c2 in enumerate(CLASSES):
        other[f"confusion_true_{c}_pred_{c2}_summed_over_seeds"] = int(sum(((yr == k) & (pred_h[s, chosen][rep_ix] == j)).sum() for s in SEEDS))

controls = {
    "word_similarity_best": {str(s): acc_best_rep for s in SEEDS},
    "body_part_parts": pc,
    "model_input_embeddings": {str(s): acc_emb_rep for s in SEEDS},
    "glove_vectors": {str(s): acc_glove_rep for s in SEEDS},
    "shuffled_labels": shuf,
}
selection = {
    "what": "Residual-stream layer used for the readout",
    "criterion": "selection_half_accuracy",
    "direction": "higher", "tie_tolerance": TIE,
    "candidates": {str(l): float(crit[l]) for l in range(L)},
    "chosen": str(chosen),
    "selection_items": [n["word"] for n in nouns if n["half"] == "selection"],
    "report_items": [n["word"] for n in nouns if n["half"] == "report"],
}
meta = json.load(open("/scratch/data/nouns_meta.json"))
setup = {"tool nouns": N, "nouns per class": meta["n_per_class"], "reporting nouns": int(rep.sum()),
         "selection nouns": int(sel.sum()), "layers": L, "carrier sentences": 12, "carrier sets": 3,
         "positive control words": len(pc_words), "label shuffles": 1000, "bootstrap resamples": 10000,
         "label margin": meta["margin"]}
write_results("nearest_body_part_accuracy", per_seed, controls, other_numbers=other, setup=setup, selection=selection)

# per-noun outputs and per-layer curves
with open("/scratch/outputs/noun_predictions.jsonl", "w") as f:
    for i, n in enumerate(nouns):
        wi = noun_ix[i]
        f.write(json.dumps(dict(word=n["word"], label=n["label"], half=n["half"],
                                hand_arm=n["hand_arm"], foot_leg=n["foot_leg"], mouth=n["mouth"],
                                pred_hidden_by_seed=[CLASSES[pred_h[s, chosen, wi]] for s in SEEDS],
                                pred_input_embeddings=CLASSES[pred_emb[wi]], pred_glove=CLASSES[pred_glove[wi]])) + "\n")
    for i, w in enumerate(pc_words):
        f.write(json.dumps(dict(word=w, label=CLASSES[pc_y[i]], half="positive_control",
                                pred_hidden_by_seed=[CLASSES[pred_h[s, chosen, idx[w]]] for s in SEEDS],
                                pred_input_embeddings=CLASSES[pred_emb[idx[w]]], pred_glove=CLASSES[pred_glove[idx[w]]])) + "\n")
json.dump(dict(acc_selection=acc_sel.tolist(), acc_report=acc_rep.tolist(), acc_positive_control=acc_pc_layers.tolist(),
               emb_report=acc_emb_rep, glove_report=acc_glove_rep, chosen=chosen, best_baseline=best_name),
          open("/scratch/outputs/layer_curves.json", "w"))
print("best baseline:", best_name, "emb", acc_emb_rep, "glove", acc_glove_rep)
print("per seed", per_seed, "pc", pc, "shuf", shuf)
print(json.dumps(other, indent=0)[:3000])
