"""Head-specific Grad-CAM++ maps on an external dataset (no retraining).

Uses the per-image output of external_eval.py, so the predictions shown are exactly those in Table 14.

  python external_gradcam.py --data <image folder used by external_eval> \
      --results <external_eval output folder, e.g. ext_results/halabja_pomegranate> \
      --runs <DSDN_out>/runs --model split_random__full__s42 --out <folder> --n 2

Outputs (in --out):
  ext_gradcam_<name>.png/.pdf       one row per image: input | symptom-head map | healthy-head map
  ext_gradcam_<name>_<class>.png    the same, one file per external class (for subfigures)
  ext_scs_per_image_<name>.csv      SCS of both maps for every external image (--no_scs to skip)
  ext_scs_by_class_<name>.csv       mean/SD per external class + Wilcoxon (symptom vs healthy) per class

Images are drawn at random (fixed --seed) inside each external class, half from correctly and half from
incorrectly handled images, so the figure is not a selection of favourable cases.
Correct = Healthy reported for healthy fruit; the mapped class for shared disease classes; any disease
for classes without a counterpart (screening only).
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsdn
from external_eval import load_image


def is_correct(r):
    if r['maps_to'] == 'other':
        return int(r['flagged_not_healthy']) == 1
    return r['pred_final'] == r['maps_to']


def pick(rows, n, rng):
    ok = [r for r in rows if is_correct(r)]; bad = [r for r in rows if not is_correct(r)]
    n_bad = min(len(bad), n // 2); n_ok = min(len(ok), n - n_bad); n_bad = min(len(bad), n - n_ok)
    sel = [ok[i] for i in sorted(rng.choice(len(ok), n_ok, replace=False))] if n_ok else []
    sel += [bad[i] for i in sorted(rng.choice(len(bad), n_bad, replace=False))] if n_bad else []
    return sel


def short(c):
    return dict(zip(dsdn.CLASSES, dsdn.SHORT)).get(c, c)


def draw(plt, cm, entries, path):
    fig, ax = plt.subplots(len(entries), 3, figsize=(9, 3.2 * len(entries)))
    ax = np.atleast_2d(ax)
    for i, (r, im, cs, ch) in enumerate(entries):
        for j, panel in enumerate([im, 0.5 * im + 0.5 * cm.inferno(cs)[..., :3], 0.5 * im + 0.5 * cm.viridis(ch)[..., :3]]):
            ax[i, j].imshow(panel); ax[i, j].axis('off')
        pdis = r.get('p_diseased') or ''
        pdis = f' | P(dis)={float(pdis):.2f}' if pdis not in ('', 'nan') else ''
        ax[i, 0].set_title(f'True: {r["external_class"]} | Pred: {short(r["pred_final"])}{pdis}', loc='left', fontsize=11)
    fig.savefig(str(path) + '.png'); fig.savefig(str(path) + '.pdf'); plt.close(fig)


def main(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    plt = dsdn.plot_style(); import matplotlib.cm as cm
    name = Path(a.results).name
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    settings = json.load(open(Path(a.results) / 'settings.json'))
    rows = list(csv.DictReader(open(Path(a.results) / f'per_image__{a.model}.csv')))
    m, _ = dsdn.load_ckpt(Path(a.runs) / a.model / 'best.pth', dev)
    assert m(torch.zeros(1, 3, dsdn.HP['img'], dsdn.HP['img']).to(dev))['h'] is not None, \
        'Grad-CAM of both heads needs a model with a binary head (full or matched_ctrl)'
    tf = dsdn.tf_eval()

    def maps(r):
        img = load_image(Path(a.data) / r['file'], settings.get('square', 'crop'))
        x = tf(img).unsqueeze(0).to(dev)
        cs, pred = dsdn.gradcampp(m, x, 'symptom'); ch, _ = dsdn.gradcampp(m, x, 'healthy')
        im = np.asarray(img.resize((dsdn.HP['img'], dsdn.HP['img']))) / 255
        return im, cs, ch, pred

    rng = np.random.default_rng(a.seed)
    classes = sorted({r['external_class'] for r in rows})
    allsel = []
    for c in classes:
        sel = pick([r for r in rows if r['external_class'] == c], a.n, rng)
        ent = []
        for r in sel:
            im, cs, ch, pred = maps(r)
            ent.append((r, im, cs, ch))
        if ent:
            draw(plt, cm, ent, out / f'ext_gradcam_{name}_{c.replace(" ", "_")}')
            allsel += ent
    draw(plt, cm, allsel, out / f'ext_gradcam_{name}')
    print('figure rows:', [(r['external_class'], r['pred_final'], 'correct' if is_correct(r) else 'wrong') for r, *_ in allsel])

    if a.no_scs:
        return
    f_img = out / f'ext_scs_per_image_{name}.csv'
    done = {r['file']: r for r in csv.DictReader(open(f_img))} if f_img.exists() else {}
    res = []
    for k, r in enumerate(rows):
        if r['file'] in done:
            res.append(done[r['file']]); continue
        _, cs, ch, _ = maps(r)
        res.append({'file': r['file'], 'external_class': r['external_class'], 'maps_to': r['maps_to'],
                    'pred_final': r['pred_final'], 'correct': int(is_correct(r)),
                    'scs_symptom': dsdn.scs(cs), 'scs_healthy': dsdn.scs(ch),
                    'map_overlap_corr': float(np.corrcoef(cs.ravel(), ch.ravel())[0, 1])})
        if (k + 1) % 200 == 0:
            dsdn.wcsv(f_img, res); print(k + 1, '/', len(rows))   # resumable
    dsdn.wcsv(f_img, res)
    from scipy.stats import wilcoxon
    tab = []
    for c in classes + ['All images']:
        rr = [x for x in res if c == 'All images' or x['external_class'] == c]
        s, h = np.array([float(x['scs_symptom']) for x in rr]), np.array([float(x['scs_healthy']) for x in rr])
        o = np.array([float(x['map_overlap_corr']) for x in rr])
        tab.append({'external_class': c, 'n': len(rr), 'scs_symptom_mean': s.mean(), 'scs_symptom_sd': s.std(),
                    'scs_healthy_mean': h.mean(), 'scs_healthy_sd': h.std(), 'map_overlap_corr_mean': np.nanmean(o),
                    'wilcoxon_p': float(wilcoxon(s, h).pvalue) if len(rr) > 10 else float('nan')})
    dsdn.wcsv(out / f'ext_scs_by_class_{name}.csv', tab)
    for t in tab:
        print(f"{t['external_class']:<24} n={t['n']:<5} SCS symptom {t['scs_symptom_mean']:.3f}  healthy "
              f"{t['scs_healthy_mean']:.3f}  overlap r={t['map_overlap_corr_mean']:.2f}  p={t['wilcoxon_p']:.2g}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--results', required=True)
    ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='split_random__full__s42')
    ap.add_argument('--n', type=int, default=2, help='images per external class in the figure')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_scs', action='store_true', help='figure only, skip SCS over all images')
    main(ap.parse_args())
