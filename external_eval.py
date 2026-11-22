"""
Zero-shot evaluation of trained models on external labelled datasets (no fine-tuning).

Each external dataset is a folder with one sub-folder per class. A JSON map tells how each
external class relates to the training classes:
    a pomegranate class name ("Healthy", "Anthracnose", "Alternaria", "Bacterial_Blight", "Cercospora"),
    or "other" for a non-healthy condition the model has never seen (e.g. sunburn, insect damage, scab).

Reported per model (see README / manuscript):
  screening   healthy specificity, non-healthy sensitivity and balanced accuracy of healthy vs not healthy
              (DSDN: binary head, P(diseased) >= tau; single-output models: five-class argmax != Healthy)
  typing      recall of each shared disease class (gated output for DSDN, argmax for single-output models)
  confusion   counts of predicted training class for every external class
  per image   all probabilities, P(diseased), gated output, latent orthogonality (FSS)
95 % confidence intervals: bootstrap over images (1,000 resamples).

Usage:
  python external_eval.py --data EXT_DIR --map map.json --runs OUT/runs --out OUT/external/NAME \
      --models split_random__full__s42,split_random__full__s123,split_random__full__s456,\
split_random__matched_ctrl__s42,split_random__plain_effb0__s42

  # crop single fruits from a YOLO-format detection dataset first (e.g. MangoFruitBD):
  python external_eval.py yolo_crop --images IMG_DIR --labels LBL_DIR --names Alternaria,Anthracnose,Healthy,Scab --out CROP_DIR
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsdn  # the training pipeline in the same folder

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_image(p, square):
    im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
    w, h = im.size
    if square == 'crop' and w != h:                       # training images are square: centre-crop to square
        s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    elif square == 'pad' and w != h:
        s = max(w, h); bg = Image.new('RGB', (s, s), (0, 0, 0)); bg.paste(im, ((s - w) // 2, (s - h) // 2)); im = bg
    if max(im.size) > 1024:
        im = im.resize((1024, 1024), Image.BICUBIC)
    return im


def boot_ci(fn, n, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); r = fn(i)
        if r == r: v.append(r)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float('nan'), float('nan'))


def evaluate(a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cmap = json.load(open(a.map))
    items = []
    for c in sorted(p.name for p in Path(a.data).iterdir() if p.is_dir()):
        if c not in cmap:
            print('skipped (not in map):', c); continue
        items += [(str(p), c) for p in sorted((Path(a.data) / c).rglob('*')) if p.suffix.lower() in EXTS]
    print(len(items), 'images;', {c: sum(1 for _, x in items if x == c) for c in cmap})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tf = dsdn.tf_eval()
    imgs = []
    for p, _ in items:
        try:
            imgs.append(tf(load_image(p, a.square)))
        except Exception as e:
            imgs.append(None); print('unreadable:', p, e)
    keep = [i for i, x in enumerate(imgs) if x is not None]
    items = [items[i] for i in keep]; imgs = [imgs[i] for i in keep]
    ext_cls = np.array([c for _, c in items])
    target = np.array([cmap[c] for c in ext_cls])
    is_healthy = target == 'Healthy'
    summary = []
    for run in a.models.split(','):
        ck = Path(a.runs) / run / 'best.pth'
        if not ck.exists():
            print('missing checkpoint:', ck); continue
        m, info = dsdn.load_ckpt(ck, dev)
        P5, P2, ZS, ZH = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(imgs), 64):
                o = m(torch.stack(imgs[i:i + 64]).to(dev))
                P5.append(F.softmax(o['d'].float(), 1).cpu())
                if o['h'] is not None:
                    P2.append(F.softmax(o['h'].float(), 1).cpu()); ZS.append(o['zs'].float().cpu()); ZH.append(o['zh'].float().cpu())
        p5 = torch.cat(P5).numpy(); two = len(P2) > 0
        if two:
            p2 = torch.cat(P2).numpy(); pdis = p2[:, 0]
            fss = dsdn.per_sample_fss(torch.cat(ZS).numpy(), torch.cat(ZH).numpy())
            final = dsdn.gate(p5, pdis, a.tau)
            flagged = pdis >= a.tau
        else:
            final = p5.argmax(1); flagged = final != dsdn.HEALTHY; pdis = None; fss = None
        # screening
        def bal(idx):
            h, nh = is_healthy[idx], ~is_healthy[idx]
            if h.sum() == 0 or nh.sum() == 0: return float('nan')
            return 0.5 * ((~flagged[idx][h]).mean() + flagged[idx][nh].mean())
        idx_all = np.arange(len(items))
        row = {'model': run, 'n_images': len(items), 'tau': a.tau if two else 'argmax',
               'healthy_specificity': float((~flagged[is_healthy]).mean()) if is_healthy.any() else float('nan'),
               'non_healthy_sensitivity': float(flagged[~is_healthy].mean()) if (~is_healthy).any() else float('nan'),
               'screening_balanced_acc': float(bal(idx_all))}
        lo, hi = boot_ci(bal, len(items)); row['screening_bal_acc_CI'] = f'[{lo:.3f}, {hi:.3f}]'
        # typing recall for shared disease classes
        for c in dsdn.CLASSES:
            if c == 'Healthy': continue
            sel = target == c
            if sel.any():
                k = dsdn.CLASSES.index(c)
                row[f'recall_{c}'] = float((final[sel] == k).mean())
                lo, hi = boot_ci(lambda i: float((final[sel][i] == k).mean()), int(sel.sum()))
                row[f'recall_{c}_CI'] = f'[{lo:.3f}, {hi:.3f}]'
                row[f'n_{c}'] = int(sel.sum())
        if two:
            row['FSS_mean'] = float(fss.mean())
        summary.append(row); print(row)
        # confusion: external class -> predicted training class
        conf = [{'external_class': c, 'maps_to': cmap[c], 'n': int((ext_cls == c).sum()),
                 **{f'pred_{k}': int(((ext_cls == c) & (final == j)).sum()) for j, k in enumerate(dsdn.CLASSES)}}
                for c in sorted(set(ext_cls))]
        dsdn.wcsv(out / f'confusion__{run}.csv', conf)
        rows = []
        for j, (p, c) in enumerate(items):
            r = {'file': str(Path(p).relative_to(a.data)), 'external_class': c, 'maps_to': cmap[c],
                 'pred_final': dsdn.CLASSES[int(final[j])], 'flagged_not_healthy': int(flagged[j]),
                 **{f'p_{k}': float(p5[j, i]) for i, k in enumerate(dsdn.CLASSES)}}
            if two:
                r['p_diseased'] = float(pdis[j]); r['fss'] = float(fss[j])
            rows.append(r)
        dsdn.wcsv(out / f'per_image__{run}.csv', rows)
        del m
    dsdn.wcsv(out / 'summary.csv', summary)
    json.dump({'data': str(a.data), 'map': cmap, 'square': a.square, 'tau': a.tau}, open(out / 'settings.json', 'w'), indent=1)
    print('written to', out)


def yolo_crop(a):
    """Crop every labelled object of a YOLO-format dataset into <out>/<class>/ (20 % margin).
    Boxes are applied to the image as stored (YOLO exports carry no EXIF rotation; images with an
    EXIF orientation tag are reported). Polygon lines (class x1 y1 x2 y2 ...) are converted to boxes."""
    names = a.names.split(','); n = 0; skipped_small = 0; polys = 0; oriented = 0; per = {}
    min_px = a.min_px
    for lp in sorted(Path(a.labels).rglob('*.txt')):
        stem = lp.stem
        ip = next((q for q in Path(a.images).rglob(stem + '.*') if q.suffix.lower() in EXTS), None)
        if ip is None: continue
        im = Image.open(ip)
        try:
            is_rot = (im.getexif() or {}).get(274, 1) not in (1, None)
        except Exception:
            is_rot = False
        if a.only_oriented and not is_rot:
            continue
        if is_rot:
            oriented += 1
            if a.oriented == 'skip':
                continue
            if a.oriented == 'transpose':
                im = ImageOps.exif_transpose(im)
        im = im.convert('RGB'); W, H = im.size
        for k, line in enumerate(open(lp).read().split('\n')):
            v = line.split()
            if len(v) < 5: continue
            try:
                c = int(float(v[0])); vals = list(map(float, v[1:]))
            except ValueError:
                continue
            if len(vals) == 4:
                x, y, w, h = vals
            else:                                   # polygon -> bounding box
                xs, ys = vals[0::2], vals[1::2]; polys += 1
                x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2; w, h = max(xs) - min(xs), max(ys) - min(ys)
            w, h = w * 1.2, h * 1.2
            box = (max(0, (x - w / 2) * W), max(0, (y - h / 2) * H), min(W, (x + w / 2) * W), min(H, (y + h / 2) * H))
            if box[2] - box[0] < min_px or box[3] - box[1] < min_px:
                skipped_small += 1; continue
            od = Path(a.out) / names[c]; od.mkdir(parents=True, exist_ok=True)
            split = lp.parent.name
            im.crop(box).save(od / f'{split}__{stem}__{k}.jpg', quality=95); n += 1
            per[names[c]] = per.get(names[c], 0) + 1
    print(f'{n} crops written to {a.out}; per class {per}; polygons converted {polys}; '
          f'skipped (< {min_px} px) {skipped_small}; images with EXIF orientation tag {oriented} '
          f'(handled: {a.oriented})')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'yolo_crop':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--images'); ap.add_argument('--labels')
        ap.add_argument('--names'); ap.add_argument('--out'); ap.add_argument('--min_px', type=int, default=64)
        ap.add_argument('--oriented', default='skip', choices=['skip', 'transpose', 'raw'],
                        help='images with an EXIF rotation tag: skip (default), rotate before cropping, or crop as stored')
        ap.add_argument('--only_oriented', action='store_true', help='crop only the EXIF-rotated images (for checking)')
        yolo_crop(ap.parse_args())
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument('--data', required=True); ap.add_argument('--map', required=True)
        ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
        ap.add_argument('--models', default='split_random__full__s42,split_random__full__s123,split_random__full__s456,'
                                           'split_random__matched_ctrl__s42,split_random__plain_effb0__s42')
        ap.add_argument('--tau', type=float, default=0.5)
        ap.add_argument('--square', default='crop', choices=['crop', 'pad', 'none'])
        evaluate(ap.parse_args())
