#!/usr/bin/env python3
"""
cho_classifier_perfreq.py  --  PER-FREQUENCY classifier (companion file).
Scores each frequency (3, 5, 11 GHz) SEPARATELY instead of pooling, and checks
whether a model trained at one frequency works at another. No charts yet.
Put in the SAME folder as cho_classifier.py, then:
    cd /scratch/eeduo ; python cho_classifier_perfreq.py
Outputs: perfreq_results.csv  and  perfreq_crossfreq.csv
"""
import os, sys
import numpy as np, pandas as pd
import cho_classifier as cc          # reuse the tested core (same folder)

DATA_ROOT = cc.DATA_ROOT
if len(sys.argv) > 1:
    DATA_ROOT = sys.argv[1]


def load_events():
    if os.path.exists("events_all.csv"):
        print("(using events_all.csv)")
        return pd.read_csv("events_all.csv")
    print("(building the table fresh)")
    return cc.build_table(DATA_ROOT)


def multiclass(df):
    if df.cell_type.nunique() < 2:
        return np.nan, df.cell_type.nunique()
    yt, yp = cc.cv_predict(cc.Xmat(df), df.cell_type.values)
    return cc.balanced_acc(yt, yp), df.cell_type.nunique()


def normal_vs_modified(df):
    nm = df[df.role.isin(["normal", "modified"])]
    if nm.role.nunique() < 2:
        return np.nan
    yt, yp = cc.cv_predict(cc.Xmat(nm), nm.role.values)
    return cc.balanced_acc(yt, yp)


def freq_key(f):
    return int("".join(ch for ch in f if ch.isdigit()) or 0)


if __name__ == "__main__":
    ev = load_events()
    freqs = sorted(ev.freq.unique(), key=freq_key)
    rows = []
    print("\nPER-FREQUENCY (within-session, optimistic):\n")
    for f in list(freqs) + ["POOLED"]:
        sub = ev if f == "POOLED" else ev[ev.freq == f]
        a, ncl = multiclass(sub); anm = normal_vs_modified(sub); hn = cc.crossday_host_nih(sub)
        rows.append(dict(freq=f, n_events=len(sub), n_cell_types=ncl,
                         multiclass_acc=None if np.isnan(a) else round(a, 1),
                         multiclass_chance=round(100 / ncl, 1) if ncl else None,
                         normal_vs_modified=None if np.isnan(anm) else round(anm, 1),
                         honest_host_nih=None if hn is None else round(hn, 1)))
        line = (f"  {f:7s} n={len(sub):6d}  {ncl} types | cell-type "
                f"{a:.0f}% (chance {100/ncl:.0f}%) | normal-vs-modified {anm:.0f}%")
        if hn is not None:
            line += f" | HONEST Host/NIH {hn:.0f}%"
        print(line)
    pd.DataFrame(rows).to_csv("perfreq_results.csv", index=False)
    print("\nsaved perfreq_results.csv")

    print("\nCROSS-FREQUENCY  (train down the side -> test across the top, cell-type %):")
    cfm = pd.DataFrame(index=freqs, columns=freqs, dtype=float)
    for fa in freqs:
        for fb in freqs:
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
            cfm.loc[fa, fb] = round(cc.balanced_acc(B.cell_type.values, pred), 0)
    print(cfm.to_string())
    cfm.to_csv("perfreq_crossfreq.csv")
    print("\nsaved perfreq_crossfreq.csv")
    print("\n(Charts come next, once we see which numbers are worth plotting.)")