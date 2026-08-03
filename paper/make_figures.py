import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

GRAY  = "#b7bcc3"
NAVY  = "#1f3b57"
SLATE = "#708090"
INK   = "#222222"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), format="pdf")
    if os.getenv("PNG_PREVIEW"):
        fig.savefig(os.path.join(OUT, name.replace(".pdf", "_preview.png")),
                    format="png")
    plt.close(fig)
    print("wrote", name)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Fig 1: held-out CE across methods (mean + seed range where available).
# No reference lines crossing the bars; annotations live in empty space.
# ---------------------------------------------------------------------------
methods = [
    "Base\n250",
    "Balanced\ninit",
    "Zero-\ncenter",
    "Toggle\n250",
    "Toggle\n500 (Py)",
    "Backprop\n300",
    "C++ no-\ntoggle 500",
    "C++\ntoggle 500",
]
mean = [7.501, 8.852, 7.862, 7.077, 6.434, 6.902, 6.996, 6.838]
low  = [7.371, 8.761, 7.504, 7.036, 6.433, 6.902, 6.996, 6.838]
high = [7.630, 8.943, 8.221, 7.117, 6.435, 6.902, 6.996, 6.838]
colors = [GRAY, GRAY, GRAY, GRAY, NAVY, GRAY, GRAY, GRAY]
x = np.arange(len(methods))
err = [mean[i] - low[i] for i in range(len(methods))]
errh = [high[i] - mean[i] for i in range(len(methods))]

fig, ax = plt.subplots(figsize=(5.8, 3.0))
bars = ax.bar(x, mean, 0.6, yerr=[err, errh], capsize=3,
              color=colors, edgecolor="black", linewidth=0.5,
              error_kw={"elinewidth": 0.8, "capthick": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=7)
ax.set_ylabel("Held-out CE (lower is better)")
ax.set_ylim(6.0, 9.9)
ax.set_yticks(np.arange(6.5, 9.8, 0.5))
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", colors=INK)

for b, v in zip(bars, mean):
    errspan = max(high[bars.index(b)] - v, v - low[bars.index(b)], 0.06)
    best_mark = " *" if bars.index(b) == 4 else ""
    ax.text(b.get_x() + b.get_width() / 2, v + errspan + 0.12,
            f"{v:.2f}{best_mark}", ha="center", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85))
ax.text(0.0, 1.0, "* best result; beats the backprop baseline (6.90)",
        transform=ax.transAxes, fontsize=7.5, color=INK,
        ha="left", va="top")
err_proxy = mlines.Line2D([], [], color="0.35", lw=1.1, marker="_",
                          markersize=7, markeredgewidth=1.1, ls="none",
                          label="error bar: min--max over 2 seeds")
ax.legend(handles=[err_proxy], loc="upper right", frameon=True,
          fontsize=7.5, fancybox=False)
despine(ax)
save(fig, "fig1_heldout_ce.pdf")


# ---------------------------------------------------------------------------
# Fig 2: ternary marginal distribution (-1/0/+1) across checkpoints.
# Labels are centered inside segments; tiny segments stay unlabeled.
# ---------------------------------------------------------------------------
stages = ["Init", "Base\n250", "Bal\n250", "Toggle\n250", "Toggle\n500",
          "Sustained\ntoggle 2000"]
neg = np.array([75.0, 98.67, 48.95, 74.44, 74.37, 48.7])
zer = np.array([25.0, 0.04, 2.12, 0.15, 0.14, 2.6])
pos = np.array([0.0, 1.29, 48.94, 25.41, 25.49, 48.7])

fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(stages))
b1 = ax.bar(x, neg, 0.6, label="-1", color="#c44e52")
b2 = ax.bar(x, zer, 0.6, bottom=neg, label="0", color="#c9c9c9")
b3 = ax.bar(x, pos, 0.6, bottom=neg + zer, label="+1", color="#55a868")
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylabel("Share of ternary weights (%)")
ax.set_ylim(0, 112)
ax.set_yticks(np.arange(0, 111, 25))
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02),
          frameon=False)

segments = [(neg, "white"), (zer, "#444444"), (pos, "white")]
for i in range(len(stages)):
    bottom = 0.0
    for seg, txtcol in segments:
        v = seg[i]
        if v >= 5.0:
            ax.text(x[i], bottom + v / 2, f"{v:.0f}%", ha="center",
                    va="center", fontsize=7.5, color=txtcol)
        bottom += v
despine(ax)
save(fig, "fig2_ternary_dist.pdf")


# ---------------------------------------------------------------------------
# Fig 3: on-device CE vs block (thr-anneal D1 vs D2), from the C++ logs.
# Reference labels live in the empty region left of x=45; legend top-right
# where the post-spike D1 curve stays below 9.5.
# ---------------------------------------------------------------------------
def parse_ce(path):
    blocks, ces = [], []
    raw = open(path, "rb").read()
    if raw[:2] == b"\xff\xfe":          # PowerShell Tee-Object writes UTF-16 LE
        raw = raw.decode("utf-16-le", errors="replace")
    else:
        raw = raw.decode("utf-8", errors="replace")
    for line in raw.splitlines():
        m = re.search(r"step\s+(\d+) \| block CE ([\d.]+)", line)
        if m:
            blocks.append(int(m.group(1)))
            ces.append(float(m.group(2)))
    return blocks, ces


D1_LOG = r"C:\Users\Noveris\AppData\Local\Temp\opencode\togD1.log"
D2_LOG = r"C:\Users\Noveris\AppData\Local\Temp\opencode\togD2.log"

d1b, d1c = parse_ce(D1_LOG)
d2b, d2c = parse_ce(D2_LOG)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(d1b, d1c, marker="o", ms=3.5, lw=1.3, color="#1a1a1a",
        label="thr-anneal from 20 (D1)")
ax.plot(d2b, d2c, marker="s", ms=3.5, lw=1.3, color="#7f8c8d", ls="--",
        label="thr-anneal from 40 (D2)")

start_line = ax.axhline(6.93, color="#b3b3b3", lw=0.9, ls=":")
end_line = ax.axhline(9.42, color="#4a4a4a", lw=0.9, ls="-.")

ax.set_xlabel("On-device blocks (1 block = 128 tokens)")
ax.set_ylabel("Block CE")
ax.set_xlim(0, 620)
ax.set_ylim(5.8, 13.5)
handles, labels = ax.get_legend_handles_labels()
handles += [start_line, end_line]
labels += ["start (zacc eval 6.93)", "sustained toggle @2000 (9.42)"]
ax.legend(handles, labels, loc="upper right", frameon=True, fancybox=False,
          fontsize=7.5)
despine(ax)
save(fig, "fig3_anneal_traj.pdf")
