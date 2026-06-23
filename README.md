# CHO RF cell classifier

Classify CHO cell types from microwave (RF) flow-cytometry "blips".

## Files
- `cho_classifier.py` builds the dataset across 3/5/11 GHz, trains a classifier,
  saves `events_all.csv` and 7 charts to `figures/`.
- `cho_classifier_perfreq.py` scores each frequency separately and cross-frequency,
  saves `perfreq_results.csv` and `perfreq_crossfreq.csv`.

## Run (on Palmetto)
    cd /scratch/eeduo
    python cho_classifier.py
    python cho_classifier_perfreq.py

## Honest status
Within-session scores look okay but overstate. The fair test (Host vs NIH on
unseen days) is about a coin flip, so the model is mostly learning the day/chip,
not the cell. 11 GHz is the most informative single frequency; pooling
frequencies hurts. Needs several more days per cell type to become trustworthy.