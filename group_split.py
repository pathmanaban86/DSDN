"""
Fruit-grouped 70/15/15 split.

Images taken within GAP_S seconds of each other in the same class folder (one fruit photographed
in a burst) form one group; groups are merged with the near-duplicate clusters found by
dataset_audit.py (pHash / embedding). A stratified group split then keeps every group in a single
subset, so no fruit and no near-duplicate appears in more than one of train / val / test.

Usage:  python group_split.py --audit audit/audit_per_image.csv --out splits
Writes: splits/split_grouped.json, splits/fruit_groups.csv
"""
import argparse, csv, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings('ignore')
GAP_S = 10          # burst gap in seconds
SEED_SEARCH = 300   # the most balanced of the first SEED_SEARCH stratified group splits is kept


def main(a):
    d = pd.read_csv(a.audit)
    d['key'] = [f'{Path(p).parent.name}/{Path(p).name}' for p in d.path]
    d['time'] = pd.to_datetime(d.time, errors='coerce')
    d = d.sort_values('time', kind='stable')
    burst = []
    for c, g in d.groupby('class'):
        k = (g.time.diff().dt.total_seconds() > GAP_S).cumsum()
        burst.append(c + '_' + k.astype(str))
    d['burst'] = pd.concat(burst)
    # union of capture bursts and near-duplicate clusters
    par = {}
    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for b, c in zip(d.burst, d.dup_cluster):
        par[find('b' + str(b))] = find('c' + str(c))
    g = pd.factorize(pd.Series([find('b' + str(b)) for b in d.burst]))[0]
    y = d.label.values
    best = None
    for seed in range(SEED_SEARCH):
        fo = np.zeros(len(d), int)
        for k, (_, te) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=seed).split(d, y, g)):
            fo[te] = k
        sub = np.where(fo < 14, 'train', np.where(fo < 17, 'val', 'test'))
        tab = pd.crosstab(d['class'], sub); frac = tab.div(tab.sum(axis=1), axis=0)
        err = ((frac[['train', 'val', 'test']] - [.7, .15, .15]) ** 2).values.sum()
        if best is None or err < best[0]:
            best = (err, seed, sub)
    _, seed, sub = best
    d['subset'] = sub; d['fruit_group'] = g
    assert d.groupby('fruit_group').subset.nunique().max() == 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({k: d.key[d.subset == k].tolist() for k in ['train', 'val', 'test']},
              open(out / 'split_grouped.json', 'w'), indent=0)
    d[['key', 'class', 'time', 'day', 'session', 'fruit_group', 'subset']].rename(
        columns={'time': 'capture_time', 'day': 'capture_day'}).to_csv(out / 'fruit_groups.csv', index=False)
    print(f'selected seed {seed}; groups {g.max() + 1}')
    print(pd.crosstab(d['class'], d.subset)[['train', 'val', 'test']])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', default='audit/audit_per_image.csv'); ap.add_argument('--out', default='splits')
    main(ap.parse_args())
