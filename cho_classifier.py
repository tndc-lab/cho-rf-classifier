#!/usr/bin/env python3
"""
cho_classifier.py  --  multi-frequency CHO cell-type classifier (self-contained).
Builds one labelled dataset across 3, 5, 11 GHz, trains a classifier, prints
honest scores, and SAVES 7 CHARTS to figures/.
Run:  cd /scratch/eeduo ; python cho_classifier.py
Deps: numpy, scipy, pandas, matplotlib.
"""
import os, re, glob, sys
from datetime import datetime
import numpy as np, pandas as pd, scipy.io as sio

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

DATA_ROOT = "/scratch/eeduo/CHO_Bulk"
if len(sys.argv) > 1:
    DATA_ROOT = sys.argv[1]
USABLE_FREQS = {"3GHz", "5GHz", "11GHz"}
CHANNELS = ["s11m", "s21m", "s11a", "s21a"]
STATS = ["ptp", "max", "min", "absarea", "energy", "width"]
WINDOW = 30
FEATS = [f"{st}_{c}" for c in CHANNELS for st in STATS]
PURPLE, GRAY, TEAL = "#7F77DD", "#B4B2A9", "#1D9E75"


def infer_role(ct):
    c = ct.lower()
    if any(k in c for k in ("vrc01", "trastuzumab")):
        return "modified"
    if "host" in c or "cho-s" in c or "cho s" in c or c.strip().lower() == "chozn" or "pf" in c:
        return "normal"
    return "clone"


def canon(name):
    n = re.sub(r"^10um\s+", "", name).strip(); n = re.sub(r"\s+", " ", n)
    if n.upper() in ("CHOZN", "CHOZN HOST"):
        return "CHOZN Host"
    if n.upper() == "CHO S":
        return "CHO-S"
    return n


def parse_log(p):
    t = open(p, errors="ignore").read()
    dut = re.search(r"Device Under Test\s+(.+)", t)
    lw = re.search(r"Microstrip Line Width\s*=\s*(\S+)", t)
    dt = re.search(r"Acquisition Start Time\s*=\s*(\d{2}-[A-Za-z]{3}-\d{4})", t)
    s = dut.group(1).strip() if dut else "? in ?"
    samp, _, dev = s.rpartition(" in "); samp = samp or s
    is_bead = bool(re.search(r"(^|\W)ps(\W|$)|10um\s*ps", samp.lower()))
    return dict(cell_type=("PS beads" if is_bead else canon(samp)), is_bead=is_bead,
                chip=f"{(dev or '?').strip()} | {lw.group(1) if lw else '?'}",
                date=datetime.strptime(dt.group(1), "%d-%b-%Y").date().isoformat() if dt else "")


def feats(d, meta):
    P = sio.loadmat(os.path.join(d, "processed_data.mat"))["processedData"]
    tms = P[P[:, 10] == 1][:, 4]
    dd = sio.loadmat(os.path.join(d, "detrend_data.mat"))
    t = dd["t"].ravel(); det = dd["detrended"]
    ch = {c: det[c][0, 0].ravel() for c in CHANNELS}
    rows = []
    for tm in tms:
        i = int(np.argmin(np.abs(t - tm))); lo, hi = max(0, i - WINDOW), min(len(t), i + WINDOW + 1)
        f = []
        for c in CHANNELS:
            sg = ch[c][lo:hi] - np.median(ch[c][lo:hi]); a = np.abs(sg); mx = a.max() if a.size else 0
            f += [sg.max() - sg.min(), sg.max(), sg.min(), a.sum(),
                  float(np.sqrt((sg ** 2).sum())), float((a > 0.5 * mx).sum())]
        rows.append(f)
    df = pd.DataFrame(rows, columns=FEATS)
    for k, v in meta.items():
        df[k] = v
    return df


def build_table(root):
    frames = []
    for pm in glob.glob(os.path.join(root, "**", "processed_data.mat"), recursive=True):
        d = os.path.dirname(pm)
        fm = re.search(r"(\d+)\s*GHz", d, re.I); freq = fm.group(0).replace(" ", "") if fm else None
        if freq not in USABLE_FREQS:
            continue
        lp = os.path.join(d, "Experiment_Log.txt")
        if not os.path.exists(lp):
            continue
        try:
            meta = parse_log(lp)
        except Exception:
            continue
        if meta["is_bead"]:
            continue
        meta.pop("is_bead")
        meta["role"] = infer_role(meta["cell_type"]); meta["freq"] = freq
        meta["session"] = os.path.relpath(d, root).split(os.sep)[0]
        try:
            df = feats(d, meta)
        except Exception as e:
            print("  (skipped)", d, "->", e); continue
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FEATS)


def Xmat(df):
    X = df[FEATS].to_numpy(float)
    return np.sign(X) * np.log1p(np.abs(X))


def lda_fit(X, y, reg=1e-1):
    cls = np.unique(y); mu = {}; Sw = np.zeros((X.shape[1],) * 2); n = 0
    for k in cls:
        Xk = X[y == k]; mu[k] = Xk.mean(0)
        if len(Xk) > 1:
            Sw += np.cov(Xk, rowvar=False) * (len(Xk) - 1); n += len(Xk) - 1
    Sw = Sw / max(n, 1) + reg * np.eye(X.shape[1]); Si = np.linalg.inv(Sw)
    return cls, {k: Si @ mu[k] for k in cls}, {k: -0.5 * mu[k] @ Si @ mu[k] for k in cls}


def lda_pred(X, cls, W, b):
    return cls[np.column_stack([X @ W[k] + b[k] for k in cls]).argmax(1)]


def cv_predict(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); folds = np.array_split(idx, k)
    yt, yp = [], []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        cls, W, b = lda_fit((X[tr] - mu) / sd, y[tr])
        yt.append(y[te]); yp.append(lda_pred((X[te] - mu) / sd, cls, W, b))
    return np.concatenate(yt), np.concatenate(yp)


def balanced_acc(yt, yp):
    return 100 * np.mean([np.mean(yp[yt == c] == c) for c in np.unique(yt)])


def crossday_host_nih(ev):
    H = ev[ev.cell_type == "CHOZN Host"]; N = ev[ev.cell_type == "CHO NIH"]
    hd, nd = sorted(H.date.unique()), sorted(N.date.unique())
    if len(hd) < 2 or len(nd) < 2:
        return None
    m = lambda df, dt: Xmat(df[df.date == dt])
    def holdout(trH, trN, teH, teN):
        Z = np.vstack([trH, trN]); yy = np.r_[np.zeros(len(trH)), np.ones(len(trN))]
        mu, sd = Z.mean(0), Z.std(0) + 1e-9; cls, W, b = lda_fit((Z - mu) / sd, yy)
        pH = lda_pred((teH - mu) / sd, cls, W, b); pN = lda_pred((teN - mu) / sd, cls, W, b)
        return 50 * ((pH == 0).mean() + (pN == 1).mean())
    accs = [holdout(m(H, h), m(N, n), m(H, [x for x in hd if x != h][0]), m(N, [x for x in nd if x != n][0]))
            for h in hd[:2] for n in nd[:2]]
    return float(np.mean(accs))


def fig_counts(ev, path):
    ct = pd.crosstab(ev.cell_type, ev.freq)
    plt.figure(figsize=(6, 5)); plt.imshow(ct.values, cmap="Greens", aspect="auto")
    plt.colorbar(fraction=.046, label="events")
    plt.xticks(range(ct.shape[1]), ct.columns, fontsize=8)
    plt.yticks(range(ct.shape[0]), ct.index, fontsize=8)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            plt.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=7)
    plt.title("events per cell type x frequency"); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def fig_accuracy(results, path):
    labels = [r[0] for r in results]; accs = [r[1] for r in results]; ch = [r[2] for r in results]
    y = np.arange(len(results))
    plt.figure(figsize=(7, 3.6))
    plt.barh(y, ch, color=GRAY, label="what guessing gives")
    plt.barh(y, [a - c for a, c in zip(accs, ch)], left=ch, color=PURPLE, label="model's extra skill")
    for i, a in enumerate(accs):
        plt.text(a + 1, i, f"{a:.0f}%", va="center", fontsize=9)
    plt.yticks(y, labels, fontsize=9); plt.xlim(0, 100); plt.xlabel("balanced accuracy %")
    plt.legend(fontsize=8, loc="lower right"); plt.title("Skill vs guessing")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def fig_confusion(yt, yp, path):
    labs = sorted(set(yt)); ix = {l: i for i, l in enumerate(labs)}
    M = np.zeros((len(labs), len(labs)))
    for a, b in zip(yt, yp):
        M[ix[a], ix[b]] += 1
    Mn = M / M.sum(1, keepdims=True).clip(min=1)
    plt.figure(figsize=(7.5, 6.2)); plt.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(fraction=.046, label="row %")
    plt.xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labs)), labs, fontsize=8)
    for i in range(len(labs)):
        for j in range(len(labs)):
            plt.text(j, i, f"{Mn[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                     color="white" if Mn[i, j] > .5 else "black")
    plt.ylabel("true cell type"); plt.xlabel("predicted")
    plt.title("Confusion matrix (cross-validated, row %)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def fig_recall(yt, yp, path):
    labs = sorted(set(yt)); rec = [np.mean(yp[yt == l] == l) * 100 for l in labs]
    plt.figure(figsize=(7, 4)); plt.barh(labs, rec, color=PURPLE)
    plt.xlabel("recall % (cross-validated)"); plt.xlim(0, 100)
    plt.title("How often each cell type is correctly caught")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def fig_pca_by(ev, col, path, title, max_pts=6000):
    df = ev.sample(min(max_pts, len(ev)), random_state=0) if len(ev) > max_pts else ev
    X = Xmat(df); Z = (X - X.mean(0)) / (X.std(0) + 1e-9); Z = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False); P = Z @ Vt.T[:, :2]
    cats = df[col].astype(str).values; uniq = sorted(set(cats))
    cmap = plt.cm.tab10 if len(uniq) <= 10 else plt.cm.tab20
    plt.figure(figsize=(7.5, 5.5))
    for i, c in enumerate(uniq):
        mm = cats == c
        plt.scatter(P[mm, 0], P[mm, 1], s=6, alpha=.4, color=cmap(i % cmap.N), label=c)
    plt.legend(fontsize=7, markerscale=2, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    xlo, xhi = np.percentile(P[:, 0], [1, 99]); ylo, yhi = np.percentile(P[:, 1], [1, 99])
    px, py = 0.1 * (xhi - xlo) + 1e-9, 0.1 * (yhi - ylo) + 1e-9
    plt.xlim(xlo - px, xhi + px); plt.ylim(ylo - py, yhi + py)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def make_charts(ev, results, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    p = lambda n: os.path.join(outdir, n)
    fig_counts(ev, p("01_event_counts.png"))
    fig_accuracy(results, p("02_skill_vs_guessing.png"))
    yt, yp = cv_predict(Xmat(ev), ev.cell_type.values)
    fig_confusion(yt, yp, p("03_confusion_matrix.png"))
    fig_recall(yt, yp, p("04_per_class_recall.png"))
    fig_pca_by(ev, "cell_type", p("05_pca_by_celltype.png"), "Cells in 2-D, coloured by CELL TYPE")
    fig_pca_by(ev, "chip", p("06_pca_by_chip.png"), "Same cells, coloured by CHIP (the confound)")
    fig_pca_by(ev, "freq", p("07_pca_by_frequency.png"), "Same cells, coloured by FREQUENCY")
    print(f"\nSaved 7 charts to {outdir}/")


if __name__ == "__main__":
    print("Building dataset from", DATA_ROOT, "...\n")
    ev = build_table(DATA_ROOT)
    ev.to_csv("events_all.csv", index=False)
    print(f"\n{len(ev):,} cell events across {ev.cell_type.nunique()} cell types "
          f"and {ev.freq.nunique()} frequencies  ->  saved events_all.csv")
    print("\nevents by cell type x frequency:")
    print(pd.crosstab(ev.cell_type, ev.freq).to_string())

    nc = ev.cell_type.nunique()
    yt, yp = cv_predict(Xmat(ev), ev.cell_type.values); acc_multi = balanced_acc(yt, yp)
    nm = ev[ev.role.isin(["normal", "modified"])]
    yt2, yp2 = cv_predict(Xmat(nm), nm.role.values); acc_nm = balanced_acc(yt2, yp2)
    hn = crossday_host_nih(ev)

    print("\n=== WITHIN-SESSION accuracy (OPTIMISTIC: mixes days, will overstate) ===")
    print(f"  guess the cell type ({nc} classes): {acc_multi:.0f}%   (chance ~{100/nc:.0f}%)")
    print(f"  normal vs modified           : {acc_nm:.0f}%   (chance 50%)")
    print("\n=== HONEST check: Host vs NIH tested on UNSEEN days ===")
    print(f"  {hn:.0f}%   (50% = chance)" if hn is not None else "  (need 2+ days of Host and NIH)")

    results = [("guess cell type", acc_multi, 100 / nc), ("normal vs modified", acc_nm, 50.0)]
    if hn is not None:
        results.append(("fair test: new days", hn, 50.0))
    if HAVE_MPL:
        make_charts(ev, results)
    else:
        print("\n(matplotlib not found -- charts skipped)")
    print("\nNOTE: within-session looks good but overstates; the unseen-day number is the honest one.")