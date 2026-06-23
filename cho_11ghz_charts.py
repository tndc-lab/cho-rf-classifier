#!/usr/bin/env python3
"""
cho_11ghz_charts.py  --  11 GHz-focused classifier + per-frequency charts.
Companion to cho_classifier.py (keep in the same folder).
Run:  cd /scratch/eeduo ; python cho_11ghz_charts.py
Reads events_all.csv. Writes figures_11ghz/ and figures_perfreq/.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cho_classifier as cc          # reuse the tested core + chart helpers

PURPLE, GRAY, TEAL = cc.PURPLE, cc.GRAY, "#1D9E75"


def load():
    if os.path.exists("events_all.csv"):
        return pd.read_csv("events_all.csv")
    return cc.build_table(cc.DATA_ROOT)


def freq_key(f):
    return int("".join(c for c in f if c.isdigit()) or 0)


def fig_perfreq_accuracy(ev, path):
    freqs = sorted(ev.freq.unique(), key=freq_key)
    multi, chance, nm = [], [], []
    for f in freqs:
        sub = ev[ev.freq == f]
        yt, yp = cc.cv_predict(cc.Xmat(sub), sub.cell_type.values)
        multi.append(cc.balanced_acc(yt, yp)); chance.append(100 / sub.cell_type.nunique())
        s = sub[sub.role.isin(["normal", "modified"])]
        if s.role.nunique() == 2:
            yt2, yp2 = cc.cv_predict(cc.Xmat(s), s.role.values); nm.append(cc.balanced_acc(yt2, yp2))
        else:
            nm.append(np.nan)
    x = np.arange(len(freqs)); w = 0.35
    plt.figure(figsize=(7, 4.3))
    plt.bar(x - w / 2, multi, w, color=PURPLE, label="guess cell type")
    plt.bar(x + w / 2, nm, w, color=TEAL, label="normal vs modified")
    plt.plot(x - w / 2, chance, "k_", ms=16, label="chance (cell type)")
    plt.axhline(50, color="gray", ls=":", lw=.8, label="chance (normal vs modified)")
    for i, (a, b) in enumerate(zip(multi, nm)):
        plt.text(i - w / 2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        if not np.isnan(b):
            plt.text(i + w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    plt.xticks(x, freqs); plt.ylim(0, 100); plt.ylabel("balanced accuracy %")
    plt.title("Accuracy by frequency (each scored on its own)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def fig_crossfreq(ev, path):
    freqs = sorted(ev.freq.unique(), key=freq_key)
    M = np.full((len(freqs), len(freqs)), np.nan)
    for i, fa in enumerate(freqs):
        for j, fb in enumerate(freqs):
            if fa == fb:
                continue
            A, B = ev[ev.freq == fa], ev[ev.freq == fb]
            common = sorted(set(A.cell_type) & set(B.cell_type))
            if len(common) < 2:
                continue
            A = A[A.cell_type.isin(common)]; B = B[B.cell_type.isin(common)]
            Xtr = cc.Xmat(A); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            cls, W, b = cc.lda_fit((Xtr - mu) / sd, A.cell_type.values)
            pred = cc.lda_pred((cc.Xmat(B) - mu) / sd, cls, W, b)
            M[i, j] = cc.balanced_acc(B.cell_type.values, pred)
    vmax = max(40.0, np.nanmax(M)) if np.isfinite(M).any() else 40.0
    plt.figure(figsize=(5.6, 4.7)); plt.imshow(M, cmap="Purples", vmin=0, vmax=vmax)
    plt.colorbar(fraction=.046, label="accuracy %")
    plt.xticks(range(len(freqs)), freqs); plt.yticks(range(len(freqs)), freqs)
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9)
    plt.xlabel("tested at"); plt.ylabel("trained at")
    plt.title("Train at one frequency, test at another\n(low = the frequencies are not interchangeable)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


if __name__ == "__main__":
    ev = load()

    e11 = ev[ev.freq == "11GHz"]
    os.makedirs("figures_11ghz", exist_ok=True)
    yt, yp = cc.cv_predict(cc.Xmat(e11), e11.cell_type.values)
    acc = cc.balanced_acc(yt, yp); nc = e11.cell_type.nunique()
    nm = e11[e11.role.isin(["normal", "modified"])]
    yt2, yp2 = cc.cv_predict(cc.Xmat(nm), nm.role.values); accnm = cc.balanced_acc(yt2, yp2)
    hn = cc.crossday_host_nih(e11)
    msg = f"11 GHz only ({len(e11):,} events): cell-type {acc:.0f}% (chance {100/nc:.0f}%) | normal-vs-modified {accnm:.0f}%"
    msg += f" | HONEST Host/NIH {hn:.0f}%" if hn is not None else "  (no Host/NIH cross-day here)"
    print(msg)
    cc.fig_confusion(yt, yp, "figures_11ghz/confusion_11ghz.png")
    cc.fig_recall(yt, yp, "figures_11ghz/recall_11ghz.png")
    cc.fig_pca_by(e11, "cell_type", "figures_11ghz/pca_celltype_11ghz.png", "11 GHz only, coloured by cell type")
    res = [("guess cell type", acc, 100 / nc), ("normal vs modified", accnm, 50.0)]
    if hn is not None:
        res.append(("fair test: new days", hn, 50.0))
    cc.fig_accuracy(res, "figures_11ghz/skill_vs_guessing_11ghz.png")
    print("saved figures_11ghz/  (4 charts)")

    os.makedirs("figures_perfreq", exist_ok=True)
    fig_perfreq_accuracy(ev, "figures_perfreq/accuracy_by_frequency.png")
    fig_crossfreq(ev, "figures_perfreq/crossfrequency_heatmap.png")
    print("saved figures_perfreq/  (2 charts)")