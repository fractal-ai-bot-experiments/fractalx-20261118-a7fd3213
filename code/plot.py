import json
import numpy as np
import fractal_plot as fp

c = json.load(open("/scratch/outputs/layer_curves.json"))
rep = np.array(c["acc_report"]); sel = np.array(c["acc_selection"])
L = rep.shape[1]; x = np.arange(L)
fig, ax = fp.figure("double")
for s in range(rep.shape[0]):
    ax.plot(x, rep[s], color="0.75", lw=0.6, marker=None)
ax.plot(x, rep.mean(0), color=fp.COLORS[0] if hasattr(fp, "COLORS") else "C0", marker="o", ms=2.5, lw=1.2)
ax.plot(x, sel.mean(0), color="C1", ls="--", marker="s", ms=2, lw=1)
ax.axhline(c["glove_report"], color="k", ls=":", lw=1)
ax.axhline(c["emb_report"], color="C2", ls="-.", lw=1)
ax.axhline(1 / 3, color="0.5", ls=(0, (1, 3)), lw=1)
ax.axvline(c["chosen"], color="0.6", lw=0.6)
ax.text(13, 0.78, "hidden state, reporting half (mean; grey = each carrier set)", color="C0", fontsize=7)
ax.text(13, 0.72, "hidden state, selection half (mean)", color="C1", fontsize=7)
ax.text(0, c["glove_report"] + 0.01, "GloVe, reporting half", fontsize=7, va="bottom")
ax.text(5, c["emb_report"] - 0.01, "input embeddings, reporting half", color="C2", fontsize=7, va="top")
ax.text(L - 1, 1 / 3 - 0.01, "chance (1/3)", fontsize=7, ha="right", va="top", color="0.4")
ax.text(c["chosen"] + 0.3, 0.98, "chosen layer", fontsize=7, va="top", color="0.4")
ax.set_xlabel("Layer (residual stream after block)")
ax.set_ylabel("Nearest-body-part accuracy (fraction)")
ax.set_ylim(0, 1)
fp.save(fig, "accuracy_by_layer")
