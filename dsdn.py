"""
DSDN: dual-projection-head network for pomegranate fruit health screening and disease typing.
Single-file pipeline (PyTorch + timm), tested on Google Colab with an NVIDIA T4.

Commands (python dsdn.py <command> ...; see README.md for the full run order)
  data      cache, split_random, split_grouped, cv_splits
  check     pilot
  train     train  (--config full | no_ortho | no_recon | matched_ctrl | no_healthy | no_disentangle |
                    plain_effb0 | bl_* baselines | sens_* loss-weight variants)
  analyse   probe, scs, fig9, export_annot, point, farms, dupcheck, xspecies
  report    summarize, stats, benchmark, figures, fig2
Every command writes CSV/JSON/PNG/PDF outputs under --out.
"""
import argparse, csv, json, math, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ExifTags
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']
SHORT = ['Alt.', 'Anth.', 'B. blight', 'Cerc.', 'Healthy']
HEALTHY = CLASSES.index('Healthy')
DIS = [i for i in range(5) if i != HEALTHY]
# Binary convention (fixed in this code): 0 = diseased, 1 = healthy
EXTS = {'.jpg', '.jpeg', '.png'}
CB = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']

# Loss weights (Table 3) and ablation configurations (Table 5 + matched-capacity control)
# two_heads, alpha (L_cls), delta (L_bin), beta (L_recon), gamma (L_ortho)
CONFIGS = {
    'full':           dict(model='dsdn', two=True,  a=0.5, d=0.2, b=0.1, g=0.2),
    'no_ortho':       dict(model='dsdn', two=True,  a=0.5, d=0.2, b=0.1, g=0.0),
    'no_recon':       dict(model='dsdn', two=True,  a=0.5, d=0.2, b=0.0, g=0.2),
    'matched_ctrl':   dict(model='dsdn', two=True,  a=0.5, d=0.2, b=0.0, g=0.0),  # same capacity + L_bin, no disentanglement
    'no_healthy':     dict(model='dsdn', two=False, a=0.5, d=0.0, b=0.1, g=0.0),  # "w/o Healthy enc."
    'no_disentangle': dict(model='dsdn', two=False, a=0.5, d=0.0, b=0.0, g=0.0),  # "w/o Disentangle" (single head)
    'plain_effb0':    dict(model='plain', arch='efficientnet_b0'),                # direct five-class classifier
}
# Nine baselines of Table 8 (ImageNet-pretrained timm models, same data, split, augmentation, schedule, 224 px).
# Transformers / ConvNeXt use the usual fine-tuning learning rate 1e-4; CNNs use 1e-3 as DSDN.
BASELINES = {
    'bl_resnet50':        ('resnet50', 1e-3),            'bl_resnet101':      ('resnet101', 1e-3),
    'bl_densenet121':     ('densenet121', 1e-3),         'bl_efficientnet_b0': ('efficientnet_b0', 1e-3),
    'bl_efficientnet_b3': ('efficientnet_b3', 1e-3),     'bl_mobilenetv3_l':  ('mobilenetv3_large_100', 1e-3),
    'bl_vit_b16':         ('vit_base_patch16_224', 1e-4), 'bl_swin_t':        ('swin_tiny_patch4_window7_224', 1e-4),
    'bl_convnext_t':      ('convnext_tiny', 1e-4),
}
BL_NAMES = {'bl_resnet50': 'ResNet-50', 'bl_resnet101': 'ResNet-101', 'bl_densenet121': 'DenseNet-121',
            'bl_efficientnet_b0': 'EfficientNet-B0', 'bl_efficientnet_b3': 'EfficientNet-B3',
            'bl_mobilenetv3_l': 'MobileNetV3-L', 'bl_vit_b16': 'ViT-B/16', 'bl_swin_t': 'Swin-T',
            'bl_convnext_t': 'ConvNeXt-Tiny', 'full': 'DSDN'}
for k, (arch, lr) in BASELINES.items():
    CONFIGS[k] = dict(model='plain', arch=arch, lr=lr)
# Table 10 sensitivity: one coefficient changed at a time from (alpha, beta, gamma) = (0.5, 0.1, 0.2)
SENS = {'sens_a0.3': ('a', 0.3), 'sens_a0.7': ('a', 0.7), 'sens_b0.05': ('b', 0.05),
        'sens_b0.2': ('b', 0.2), 'sens_g0.1': ('g', 0.1), 'sens_g0.3': ('g', 0.3)}
for k, (p, v) in SENS.items():
    CONFIGS[k] = {**CONFIGS['full'], p: v}
HP = dict(epochs=25, lr=1e-3, wd=1e-4, bs=16, patience=10, img=224)


# ============================================================ utilities
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def key_of(p):  # class/file — stable id independent of Drive paths
    p = Path(p); return f'{p.parent.name}/{p.name}'


def list_images(data):
    items = []
    for ci, c in enumerate(CLASSES):
        d = Path(data) / c
        if not d.exists():
            d = next(p for p in Path(data).iterdir() if p.is_dir() and p.name.lower() == c.lower())
        items += [(str(p), ci) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return items


def label_of_key(k):
    return [c.lower() for c in CLASSES].index(k.split('/')[0].lower())


def wcsv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def plot_style():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 14, 'axes.grid': False, 'figure.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    return plt


# ============================================================ data
def cmd_cache(a):
    """Resized 256x256 copies of the originals.
    1) Built ON DRIVE (--cache_drive), image by image, so an interrupted build resumes where it stopped.
    2) Packed once into <cache_drive>.zip on Drive.
    3) Each new Colab session: unzipped to fast local disk (--cache), about 1 minute."""
    items = list_images(a.data); drive = Path(a.cache_drive); local = Path(a.cache)
    zip_path = drive.with_suffix('.zip')
    n_local = sum(1 for q in local.rglob('*.jpg') if not q.name.endswith('.tmp.jpg')) if local.exists() else 0
    if n_local == len(items):
        print(f'local cache complete ({n_local} images)'); return
    if not zip_path.exists():
        todo = [(p, drive / key_of(p)) for p, _ in items if not (drive / key_of(p)).exists()]
        print(f'building Drive cache: {len(items) - len(todo)} done, {len(todo)} to go')
        for i, (p, q) in enumerate(todo):
            q.parent.mkdir(parents=True, exist_ok=True); tmp = q.with_suffix('.tmp.jpg')
            ImageOps.exif_transpose(Image.open(p)).convert('RGB').resize((256, 256), Image.BICUBIC).save(tmp, 'JPEG', quality=95)
            os.replace(tmp, q)
            if i % 250 == 0: print(f'  {i}/{len(todo)}')
        for t in drive.rglob('*.tmp.jpg'): t.unlink()   # leftovers of an interrupted write
        tmpzip = zip_path.with_suffix('.tmp.zip')
        shutil.make_archive(str(tmpzip.with_suffix('')), 'zip', root_dir=drive.parent, base_dir=drive.name)
        os.replace(tmpzip, zip_path); print('Drive cache zipped ->', zip_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(local.parent / '_unz'))
    if local.exists(): shutil.rmtree(local)
    os.replace(local.parent / '_unz' / drive.name, local); shutil.rmtree(local.parent / '_unz', ignore_errors=True)
    print('local cache ready:', sum(1 for _ in local.rglob('*.jpg')), 'images')


def cmd_split_random(a):
    """Stratified 70/15/15 random split (seed fixed) — the protocol of the main experiments."""
    from sklearn.model_selection import train_test_split
    items = list_images(a.data); keys = np.array([key_of(p) for p, _ in items]); y = np.array([c for _, c in items])
    tr, tmp = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=a.seed)
    save_split({'train': keys[tr], 'val': keys[va], 'test': keys[te]}, Path(a.out) / 'splits' / 'split_random.json')


def cmd_split_grouped(a):
    """Convert split_fruitgroup.json (Drive paths, fruit/shot groups) to key format."""
    s = json.load(open(a.groups_json))
    save_split({k: [key_of(p) for p in v] for k, v in s.items()}, Path(a.out) / 'splits' / 'split_grouped.json')


def cmd_cv_splits(a):
    """5 group-aware folds over train+val of a split (test fixed) for the cross-validation outputs."""
    from sklearn.model_selection import StratifiedGroupKFold
    s = json.load(open(a.split)); pool = s['train'] + s['val']
    y = np.array([label_of_key(k) for k in pool])
    if a.groups_csv:
        g = {key_of(r.get('path') or r['key']): r['fruit_group'] for r in csv.DictReader(open(a.groups_csv))}
        groups = np.array([g[k] for k in pool])
    else:
        groups = np.arange(len(pool))
    for f, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(pool, y, groups)):
        save_split({'train': [pool[i] for i in tr], 'val': [pool[i] for i in va], 'test': s['test']},
                   Path(a.out) / 'splits' / f'{Path(a.split).stem}_cv{f}.json')


def save_split(s, path):
    if path.exists():
        print('exists, kept:', path); return
    path.parent.mkdir(parents=True, exist_ok=True)
    s = {k: [str(x) for x in v] for k, v in s.items()}
    json.dump(s, open(path, 'w'), indent=0)
    for k, v in s.items():
        cnt = np.bincount([label_of_key(x) for x in v], minlength=5)
        print(f'{path.name:28s} {k:5s} n={len(v):4d}', dict(zip(SHORT, cnt.tolist())))


def tf_train():
    return transforms.Compose([
        transforms.RandomResizedCrop(HP['img'], scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3), transforms.RandomRotation(20), transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def tf_eval():
    return transforms.Compose([transforms.Resize((HP['img'], HP['img'])), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class KeyDS(Dataset):
    def __init__(self, keys, cache, tf):
        self.k, self.cache, self.tf = list(keys), Path(cache), tf
        self.y = [label_of_key(k) for k in self.k]
    def __len__(self): return len(self.k)
    def __getitem__(self, i):
        y = self.y[i]
        x = self.tf(Image.open(self.cache / self.k[i]).convert('RGB'))
        return x, y, int(y == HEALTHY), i


def make_loader(keys, cache, train, gen_seed=None):
    g = torch.Generator(); g.manual_seed(gen_seed if gen_seed is not None else 0)
    return DataLoader(KeyDS(keys, cache, tf_train() if train else tf_eval()), batch_size=HP['bs'],
                      shuffle=train, num_workers=2, pin_memory=True, drop_last=False, generator=g)


# ============================================================ model
def backbone(pretrained):
    import timm
    return timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)


class Proj(nn.Sequential):  # Eq. (1)/(2): Linear(1280->256) -> ReLU -> Dropout -> Linear(256->128)
    def __init__(self, p=0.2):
        super().__init__(nn.Linear(1280, 256), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(256, 128))


class DSDN(nn.Module):
    """Shared EfficientNet-B0; symptom head -> 5-class; healthy head -> binary; decoder on [z_h || z_s]."""
    def __init__(self, two=True, pretrained=True):
        super().__init__()
        self.two = two
        self.encoder = backbone(pretrained)
        self.symptom_encoder = Proj()
        self.disease_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 5))
        if two:
            self.healthy_encoder = Proj()
            self.healthy_classifier = nn.Sequential(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, 2))
        self.decoder = nn.Sequential(nn.Linear(256 if two else 128, 512), nn.ReLU(inplace=True), nn.Dropout(0.0),
                                     nn.Linear(512, 1280))

    def forward(self, x):
        f = self.encoder(x); zs = self.symptom_encoder(f); d = self.disease_classifier(zs)
        if self.two:
            zh = self.healthy_encoder(f); h = self.healthy_classifier(zh)
            rec = self.decoder(torch.cat([zh, zs], 1))
        else:
            zh, h, rec = None, None, self.decoder(zs)
        return dict(d=d, h=h, zs=zs, zh=zh, rec=rec, f=f)


class Plain(nn.Module):
    def __init__(self, pretrained=True, arch='efficientnet_b0'):
        super().__init__()
        import timm
        self.encoder = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        with torch.no_grad():  # true pooled-feature size (e.g. MobileNetV3: 1280, not num_features=960)
            was = self.encoder.training; self.encoder.eval()
            dim = self.encoder(torch.zeros(1, 3, 224, 224)).shape[1]; self.encoder.train(was)
        self.fc = nn.Linear(dim, 5)
    def forward(self, x):
        f = self.encoder(x); return dict(d=self.fc(f), h=None, zs=None, zh=None, rec=None, f=f)


def build(config, pretrained=True):
    c = CONFIGS[config]
    return Plain(pretrained, c.get('arch', 'efficientnet_b0')) if c['model'] == 'plain' else DSDN(c['two'], pretrained)


def n_params(m): return sum(p.numel() for p in m.parameters())


def atomic_save(obj, path):
    """Write to a temporary file, then rename: a disconnect can never leave a half-written checkpoint."""
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = Path(str(path) + '.tmp'); json.dump(obj, open(tmp, 'w'), indent=1); os.replace(tmp, path)


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build(ck['config'], pretrained=False); m.load_state_dict(ck['state_dict']); return m.to(dev).eval(), ck


# ============================================================ train
def run_name(a): return f'{Path(a.split).stem}__{a.config}__s{a.seed}'


def cmd_train(a):
    """Resumable: state saved to Drive after every epoch (last.pth). Rerunning the same command
    continues from the last finished epoch; finished runs (metrics.json present) are skipped."""
    rd = Path(a.out) / 'runs' / run_name(a); rd.mkdir(parents=True, exist_ok=True)
    if (rd / 'metrics.json').exists() and not a.force:
        print('done, skipped:', rd.name); return
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = CONFIGS[a.config]; s = json.load(open(a.split)); epochs = a.epochs or HP['epochs']
    set_seed(a.seed)
    m = build(a.config, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=c.get('lr', HP['lr']), weight_decay=HP['wd'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    st = {'epoch': 0, 'best': -1.0, 'bad': 0, 'log': [], 'finished': False}
    if (rd / 'last.pth').exists():
        ck = torch.load(rd / 'last.pth', map_location='cpu', weights_only=False)  # RNG states must stay on CPU
        m.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler']); st = ck['state']
        random.setstate(ck['rng_py']); np.random.set_state(ck['rng_np'])
        torch.set_rng_state(ck['rng_torch'].cpu())
        if dev == 'cuda' and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng_cuda']])
        print(f'RESUMING {rd.name} after epoch {st["epoch"]} (best val acc {st["best"]:.4f})')
    va = make_loader(s['val'], a.cache, False)
    ep = st['epoch']
    while not st['finished'] and ep < epochs:
        ep += 1
        tr = make_loader(s['train'], a.cache, True, gen_seed=a.seed * 1000 + ep)  # same order/augmentation on resume
        m.train(); t0 = time.time(); agg = defaultdict(float); n = 0; corr = 0
        for x, y, hb, _ in tr:
            x, y, hb = x.to(dev, non_blocking=True), y.to(dev), hb.to(dev)
            with torch.autocast(device_type='cuda', enabled=dev == 'cuda'):
                o = m(x)
            o = {k: (v.float() if v is not None else None) for k, v in o.items()}
            zero = torch.zeros((), device=dev)
            if c['model'] == 'plain':
                Lc = F.cross_entropy(o['d'], y); Lb = Lr = Lo = zero; L = Lc
            else:
                Lc = F.cross_entropy(o['d'], y)
                Lb = F.cross_entropy(o['h'], hb) if c['two'] else zero  # hb: 0 = diseased, 1 = healthy
                Lr = F.mse_loss(o['rec'], o['f'].detach())
                Lo = (F.cosine_similarity(o['zs'], o['zh'], dim=1) ** 2).mean() if c['two'] else zero
                L = c['a'] * Lc + c['d'] * Lb + c['b'] * Lr + c['g'] * Lo
            opt.zero_grad(set_to_none=True); scaler.scale(L).backward(); scaler.step(opt); scaler.update()
            k = x.size(0); n += k; corr += (o['d'].argmax(1) == y).sum().item()
            for nm, v in [('loss_total', L), ('loss_cls', Lc), ('loss_bin', Lb), ('loss_recon', Lr), ('loss_ortho', Lo)]:
                agg[nm] += v.item() * k
        lr_used = opt.param_groups[0]['lr']; sch.step()
        vm = core_metrics(predict(m, va, dev))
        row = {'epoch': ep, 'lr': lr_used, **{k: v / n for k, v in agg.items()},
               'train_acc': corr / n, 'val_acc': vm['acc'], 'val_f1_w': vm['f1_weighted'],
               'val_bin_acc': vm.get('bin_acc', float('nan')), 'val_fss': vm.get('fss', float('nan')),
               'sec': round(time.time() - t0, 1)}
        st['log'].append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        if vm['acc'] > st['best']:
            st['best'], st['bad'] = vm['acc'], 0
            atomic_save({'state_dict': m.state_dict(), 'config': a.config, 'seed': a.seed, 'epoch': ep,
                         'split': str(a.split), 'hp': HP, 'loss_weights': c}, rd / 'best.pth')
        else:
            st['bad'] += 1
        st['epoch'] = ep
        st['finished'] = st['bad'] >= (a.patience or HP['patience']) or ep >= epochs
        wcsv(rd / 'epoch_log.csv', st['log'])
        atomic_save({'model': m.state_dict(), 'opt': opt.state_dict(), 'sched': sch.state_dict(),
                     'scaler': scaler.state_dict(), 'state': st, 'rng_py': random.getstate(),
                     'rng_np': np.random.get_state(), 'rng_torch': torch.get_rng_state().cpu(),
                     'rng_cuda': torch.cuda.get_rng_state_all() if dev == 'cuda' else None}, rd / 'last.pth')
    print(f'stopped after epoch {st["epoch"]}; best val acc {st["best"]:.4f}')
    evaluate_run(rd, a.cache, dev)
    if not a.keep_last and (rd / 'metrics.json').exists():
        (rd / 'last.pth').unlink(missing_ok=True)   # frees Drive space; best.pth is kept


@torch.no_grad()
def predict(m, loader, dev):
    m.eval(); R = defaultdict(list)
    for x, y, hb, idx in loader:
        o = m(x.to(dev))
        R['p5'].append(F.softmax(o['d'].float(), 1).cpu()); R['y'].append(y); R['idx'].append(idx)
        if o['zs'] is not None: R['zs'].append(o['zs'].float().cpu())
        if o['h'] is not None:
            R['p2'].append(F.softmax(o['h'].float(), 1).cpu()); R['zh'].append(o['zh'].float().cpu())
        R['f'].append(o['f'].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in R.items()}


def per_sample_fss(zs, zh):
    zs = zs / (np.linalg.norm(zs, axis=1, keepdims=True) + 1e-12)
    zh = zh / (np.linalg.norm(zh, axis=1, keepdims=True) + 1e-12)
    return 1 - np.abs((zs * zh).sum(1))  # Eq. (11): FSS = 1 - |cos(z_s, z_h)|, averaged over images


def core_metrics(P):
    from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score
    y, pred = P['y'], P['p5'].argmax(1)
    out = {'n': int(len(y)), 'acc': float((pred == y).mean()),
           'f1_weighted': float(f1_score(y, pred, average='weighted')),
           'f1_macro': float(f1_score(y, pred, average='macro')),
           'mcc': float(matthews_corrcoef(y, pred)), 'kappa': float(cohen_kappa_score(y, pred))}
    for i, c in enumerate(CLASSES):
        out[f'f1_{c}'] = float(f1_score(y == i, pred == i))
    if 'p2' in P:
        hb = (y == HEALTHY).astype(int); pb = P['p2'].argmax(1)
        out['bin_acc'] = float((pb == hb).mean())
        out['bin_sensitivity_diseased'] = float((pb[hb == 0] == 0).mean())
        out['bin_specificity_healthy'] = float((pb[hb == 1] == 1).mean())
        fs = per_sample_fss(P['zs'], P['zh']); out['fss'] = float(fs.mean()); out['fss_sd'] = float(fs.std())
        for i, c in enumerate(CLASSES):
            out[f'fss_{c}'] = float(fs[y == i].mean())
    return out


def gate(p5, pdis, tau):
    """Final two-stage inference rule:
       Stage 1: if P(diseased) < tau  -> 'Healthy' (five-class output suppressed)
       Stage 2: otherwise             -> argmax of the five-class head over the four disease classes."""
    dis = p5[:, DIS].argmax(1)
    return np.where(pdis < tau, HEALTHY, np.array(DIS)[dis])


def screening_metrics(y, pred):
    h, d = y == HEALTHY, y != HEALTHY
    return {'acc_end_to_end': float((pred == y).mean()),
            'healthy_FPR': float((pred[h] != HEALTHY).mean()),       # healthy fruit reported as diseased
            'diseased_FNR': float((pred[d] == HEALTHY).mean()),      # diseased fruit reported as healthy
            'typing_acc_on_diseased': float((pred[d] == y[d]).mean())}


def evaluate_run(rd, cache, dev):
    m, ck = load_ckpt(rd / 'best.pth', dev); s = json.load(open(ck['split']))
    out = {'config': ck['config'], 'seed': ck['seed'], 'best_epoch': ck['epoch'], 'params': n_params(m)}
    for subset in ['val', 'test']:
        P = predict(m, make_loader(s[subset], cache, False), dev)
        np.savez_compressed(rd / f'pred_{subset}.npz', **P, keys=np.array(s[subset])[P['idx']])
        out[subset] = core_metrics(P)
        rows = []
        for j, i in enumerate(P['idx']):
            r = {'key': s[subset][i], 'y': int(P['y'][j]), 'pred_5class': int(P['p5'][j].argmax())}
            r.update({f'p_{c}': float(P['p5'][j, k]) for k, c in enumerate(CLASSES)})
            if 'p2' in P:
                r['p_diseased'] = float(P['p2'][j, 0]); r['pred_binary'] = int(P['p2'][j].argmax())
                r['fss'] = float(per_sample_fss(P['zs'][j:j + 1], P['zh'][j:j + 1])[0])
                r['pred_gated_0.5'] = int(gate(P['p5'][j:j + 1], P['p2'][j:j + 1, 0], 0.5)[0])
                r['binary_vs_5class_disagree'] = int((r['pred_binary'] == 1) != (r['pred_5class'] == HEALTHY))
            rows.append(r)
        wcsv(rd / f'predictions_{subset}.csv', rows)
    # gated vs direct inference (threshold tau selected on validation by balanced screening accuracy)
    if 'p2' in np.load(rd / 'pred_val.npz'):
        V, T = np.load(rd / 'pred_val.npz'), np.load(rd / 'pred_test.npz')
        taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
        def bal(P, t):
            sm = screening_metrics(P['y'], gate(P['p5'], P['p2'][:, 0], t)); return 1 - (sm['healthy_FPR'] + sm['diseased_FNR']) / 2
        tau_val = float(taus[np.argmax([bal(V, t) for t in taus])])
        out['screening'] = {'tau_selected_on_val': tau_val,
                     'direct_5class_head': screening_metrics(T['y'], T['p5'].argmax(1)),
                     'gated_tau0.5': screening_metrics(T['y'], gate(T['p5'], T['p2'][:, 0], 0.5)),
                     f'gated_tau_val': screening_metrics(T['y'], gate(T['p5'], T['p2'][:, 0], tau_val)),
                     'sweep_test': {str(t): screening_metrics(T['y'], gate(T['p5'], T['p2'][:, 0], t)) for t in taus},
                     'n_disagreements_test': int(((T['p2'].argmax(1) == 1) != (T['p5'].argmax(1) == HEALTHY)).sum())}
    elif CONFIGS[ck['config']]['model'] == 'plain':
        T = np.load(rd / 'pred_test.npz'); out['screening'] = {'direct_5class_plain': screening_metrics(T['y'], T['p5'].argmax(1))}
    atomic_json(out, rd / 'metrics.json')
    print(json.dumps({k: v for k, v in out.items() if k not in ('val',)}, indent=1)[:2500])


# ============================================================ latent probing
def cmd_probe(a):
    """Linear probes on frozen latents (fit on train, score on test). Key cells:
       z_h -> disease type and z_s -> health state. Backbone f gives the upper reference."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score as bacc
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for rd in sorted((Path(a.out) / 'runs').glob('*__full__*')) + sorted((Path(a.out) / 'runs').glob('*__no_ortho__*')) \
            + sorted((Path(a.out) / 'runs').glob('*__matched_ctrl__*')):
        if not (rd / 'best.pth').exists() or not (rd / 'metrics.json').exists(): continue
        if (rd / 'probe.json').exists() and not a.force: print('done, skipped:', rd.name); continue
        m, ck = load_ckpt(rd / 'best.pth', dev); s = json.load(open(ck['split']))
        A = predict(m, make_loader(s['train'], a.cache, False), dev)
        B = predict(m, make_loader(s['test'], a.cache, False), dev)
        res = {'config': ck['config'], 'seed': ck['seed']}
        for z in ['zs', 'zh', 'f']:
            for tgt in ['disease5', 'health2', 'disease4_diseased_only']:
                ya, yb, ma, mb = A['y'], B['y'], np.ones(len(A['y']), bool), np.ones(len(B['y']), bool)
                if tgt == 'health2': ya, yb = (ya == HEALTHY).astype(int), (yb == HEALTHY).astype(int)
                if tgt == 'disease4_diseased_only': ma, mb = A['y'] != HEALTHY, B['y'] != HEALTHY
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(A[z][ma], ya[ma])
                res[f'{z}->{tgt}'] = float(bacc(yb[mb], clf.predict(B[z][mb])))
        atomic_json(res, rd / 'probe.json'); print(res)


# ============================================================ Grad-CAM++ attribution
def gradcampp(m, x, which):
    """which='symptom': gradient of the predicted disease-type logit;
       which='healthy': gradient of the 'diseased' logit of the binary head.
       Target layer: encoder.blocks[6][0].conv_pwl (320 ch, 7x7), shared by both maps."""
    layer = m.encoder.blocks[6][0].conv_pwl; store = []
    hd = layer.register_forward_hook(lambda mod, i, o: store.append(o))
    with torch.enable_grad():
        o = m(x); hd.remove()
        pred = int(o['d'].argmax(1)); score = o['d'][0, pred] if which == 'symptom' else o['h'][0, 0]
        G = torch.autograd.grad(score, store[0])[0]
    A, G = store[0].detach(), G.detach()
    al = G ** 2 / (2 * G ** 2 + A.sum((2, 3), keepdim=True) * G ** 3 + 1e-8)
    w = (al * F.relu(G)).sum((2, 3), keepdim=True)
    cam = F.interpolate(F.relu((w * A).sum(1, keepdim=True)), (HP['img'], HP['img']), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy(); cam -= cam.min(); cam /= cam.max() + 1e-8
    return cam, pred


def scs(cam, top=0.20):  # Eq. (12)
    f = cam.ravel(); t = np.percentile(f, 100 * (1 - top)); return float(f[f >= t].sum() / (f.sum() + 1e-8))


def first_run(out, split_stem, config='full', seed=42):
    rd = Path(out) / 'runs' / f'{split_stem}__{config}__s{seed}'
    assert (rd / 'metrics.json').exists(), f'run not finished: {rd}'
    return rd


def cmd_scs(a):
    """SCS for both maps on the WHOLE test set; per-image CSV + per-class table (Table 13)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rd = first_run(a.out, a.split_stem, seed=a.seed)
    if (rd / 'scs_per_image.csv').exists() and not a.force:
        rows = [{k: (float(v) if k.startswith(('scs', 'map')) else (int(v) if k in ('y', 'pred', 'correct') else v))
                 for k, v in r.items()} for r in csv.DictReader(open(rd / 'scs_per_image.csv'))]
        print('using saved scs_per_image.csv')
    else:
        rows = None
    m, ck = load_ckpt(rd / 'best.pth', dev)
    s = json.load(open(ck['split'])); tf = tf_eval()
    todo = [] if rows is not None else s['test']; rows = rows if rows is not None else []
    for k in todo:
        x = tf(Image.open(Path(a.cache) / k).convert('RGB')).unsqueeze(0).to(dev)
        cs, pred = gradcampp(m, x, 'symptom'); ch, _ = gradcampp(m, x, 'healthy')
        rows.append({'key': k, 'y': label_of_key(k), 'pred': pred, 'correct': int(pred == label_of_key(k)),
                     'scs_symptom': scs(cs), 'scs_healthy': scs(ch),
                     'map_overlap_corr': float(np.corrcoef(cs.ravel(), ch.ravel())[0, 1])})
    wcsv(rd / 'scs_per_image.csv', rows)
    tab = []
    for i, c in enumerate(CLASSES):
        r = [x for x in rows if x['y'] == i]
        tab.append({'class': c, 'n': len(r), **{f'{k}_{st}': f(np.array([x[k] for x in r]))
                    for k in ['scs_symptom', 'scs_healthy'] for st, f in [('mean', np.mean), ('sd', np.std)]}})
    r = [x for x in rows if x['y'] != HEALTHY]
    tab.append({'class': 'Disease classes (pooled images)', 'n': len(r),
                **{f'{k}_{st}': f(np.array([x[k] for x in r])) for k in ['scs_symptom', 'scs_healthy']
                   for st, f in [('mean', np.mean), ('sd', np.std)]}})
    from scipy.stats import wilcoxon
    tab[-1]['wilcoxon_p_symptom_vs_healthy'] = float(wilcoxon([x['scs_symptom'] for x in r], [x['scs_healthy'] for x in r]).pvalue)
    wcsv(Path(a.out) / 'tables' / 'table_scs_by_class.csv', tab); print(tab)


def cmd_fig9(a):
    """Dual-map figure: per class the most confident correct test image + misclassified test images."""
    plt = plot_style(); import matplotlib.cm as cm
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rd = first_run(a.out, a.split_stem, seed=a.seed); m, ck = load_ckpt(rd / 'best.pth', dev)
    pr = list(csv.DictReader(open(rd / 'predictions_test.csv')))
    chosen = []
    for i in range(5):
        ok = [r for r in pr if int(r['y']) == i and int(r['pred_5class']) == i]
        if ok: chosen.append(max(ok, key=lambda r: float(r[f'p_{CLASSES[i]}'])))
    wrong = [r for r in pr if int(r['y']) != int(r['pred_5class'])][:a.n_wrong]
    for tag, sel in [('correct', chosen), ('misclassified', wrong)]:
        if not sel: print('no', tag, 'examples'); continue
        fig, ax = plt.subplots(len(sel), 3, figsize=(9, 3.1 * len(sel))); ax = np.atleast_2d(ax)
        for r_i, r in enumerate(sel):
            img = Image.open(Path(a.cache) / r['key']).convert('RGB').resize((HP['img'], HP['img']))
            x = tf_eval()(img).unsqueeze(0).to(dev)
            cs, _ = gradcampp(m, x, 'symptom'); ch, _ = gradcampp(m, x, 'healthy'); im = np.asarray(img) / 255
            for c_i, (panel) in enumerate([im, 0.5 * im + 0.5 * cm.inferno(cs)[..., :3], 0.5 * im + 0.5 * cm.viridis(ch)[..., :3]]):
                ax[r_i, c_i].imshow(panel); ax[r_i, c_i].axis('off')
            ax[r_i, 0].set_title(f'True: {SHORT[int(r["y"])]} | Pred: {SHORT[int(r["pred_5class"])]}'
                                 f' | P(dis)={float(r.get("p_diseased", "nan")):.2f}', loc='left', fontsize=11)
        fig.savefig(Path(a.out) / f'fig9_{tag}.png'); fig.savefig(Path(a.out) / f'fig9_{tag}.pdf'); plt.close(fig)
    print('columns: input | symptom-head map (disease-type gradient) | healthy-head map (binary diseased gradient)')


def cmd_export_annot(a):
    """Copies a test subset (n per disease class + all misclassified) at 512 px for lesion annotation
    in LabelMe (pip install labelme; draw polygons labelled 'lesion'; save JSON next to each image)."""
    rd = first_run(a.out, a.split_stem, seed=a.seed); pr = list(csv.DictReader(open(rd / 'predictions_test.csv')))
    rng = random.Random(0); sel = []
    for i in DIS:
        sel += rng.sample([r['key'] for r in pr if int(r['y']) == i], min(a.n, sum(int(r['y']) == i for r in pr)))
    sel += [r['key'] for r in pr if int(r['y']) != int(r['pred_5class']) and int(r['y']) != HEALTHY]
    od = Path(a.annot); od.mkdir(parents=True, exist_ok=True)
    for k in dict.fromkeys(sel):
        Image.open(Path(a.data) / k).convert('RGB').resize((512, 512), Image.BICUBIC).save(od / k.replace('/', '__'))
    print(len(set(sel)), 'images exported to', od)


def cmd_point(a):
    """Pointing game + lesion-energy fraction against expert polygons (LabelMe JSON)."""
    from PIL import ImageDraw
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rd = first_run(a.out, a.split_stem, seed=a.seed); m, ck = load_ckpt(rd / 'best.pth', dev); rows = []
    for jf in sorted(Path(a.annot).glob('*.json')):
        J = json.load(open(jf)); k = jf.stem.replace('__', '/') + Path(J['imagePath']).suffix
        W, H = J['imageWidth'], J['imageHeight']; mk = Image.new('L', (HP['img'], HP['img']), 0); d = ImageDraw.Draw(mk)
        for sh in J['shapes']:
            if sh['label'].lower().startswith('lesion'):
                d.polygon([(px * HP['img'] / W, py * HP['img'] / H) for px, py in sh['points']], fill=1)
        mk = np.asarray(mk).astype(bool)
        if mk.sum() == 0: continue
        x = tf_eval()(Image.open(Path(a.cache) / k).convert('RGB')).unsqueeze(0).to(dev)
        cs, pred = gradcampp(m, x, 'symptom'); ch, _ = gradcampp(m, x, 'healthy')
        r = {'key': k, 'y': label_of_key(k), 'pred': pred, 'correct': int(pred == label_of_key(k)),
             'lesion_area_frac': float(mk.mean())}
        for nm, c in [('symptom', cs), ('healthy', ch)]:
            yy, xx = np.unravel_index(c.argmax(), c.shape)
            r[f'pointing_hit_{nm}'] = int(mk[yy, xx]); r[f'energy_in_lesion_{nm}'] = float(c[mk].sum() / (c.sum() + 1e-8))
        rows.append(r)
    wcsv(rd / 'pointing_per_image.csv', rows)
    tab = []
    for grp, sel in [('all', rows), ('correct', [r for r in rows if r['correct']]), ('misclassified', [r for r in rows if not r['correct']])] + \
                    [(CLASSES[i], [r for r in rows if r['y'] == i]) for i in DIS]:
        if sel:
            tab.append({'group': grp, 'n': len(sel), **{k: float(np.mean([r[k] for r in sel])) for k in
                        ['pointing_hit_symptom', 'pointing_hit_healthy', 'energy_in_lesion_symptom',
                         'energy_in_lesion_healthy', 'lesion_area_frac']}})
    wcsv(Path(a.out) / 'tables' / 'table_pointing_game.csv', tab); print(tab)
    print('Chance level: pointing hit ~ lesion_area_frac; energy-in-lesion ~ lesion_area_frac.')


# ============================================================ acquisition structure
def _gps(p):
    try:
        ex = Image.open(p)._getexif() or {}
        gi = ex.get({v: k for k, v in ExifTags.TAGS.items()}['GPSInfo'])
        if not gi: return None
        g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gi.items()}
        dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat, lon = dms(g['GPSLatitude']), dms(g['GPSLongitude'])
        return (-lat if g.get('GPSLatitudeRef') == 'S' else lat, -lon if g.get('GPSLongitudeRef') == 'W' else lon)
    except Exception:
        return None


def cmd_farms(a):
    """Locates acquisition sites from EXIF GPS (km-scale clustering) and reports their distribution
    over classes and over train/val/test for each split file."""
    from sklearn.cluster import AgglomerativeClustering
    items = list_images(a.data); rows = []
    for p, c in items:
        g = _gps(p); rows.append({'key': key_of(p), 'class': CLASSES[c], 'lat': g[0] if g else None, 'lon': g[1] if g else None})
    has = [r for r in rows if r['lat'] is not None]
    if len(has) < 2:
        print('fewer than 2 images with GPS; site analysis skipped'); has = []
    X = np.radians([[r['lat'], r['lon']] for r in has]) * 6371.0  # ~km
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=a.site_km, linkage='single').fit_predict(X) if has else []
    for r, l in zip(has, lab): r['site'] = f'site_{l}'
    for r in rows: r.setdefault('site', 'no_GPS')
    cent = defaultdict(list)
    for r in has: cent[r['site']].append((r['lat'], r['lon']))
    print({k: len(v) for k, v in cent.items()})   # site sizes only; coordinates are never written
    wcsv(Path(a.out) / 'tables' / 'site_per_image.csv', [{'key': r['key'], 'class': r['class'], 'site': r['site']} for r in rows])
    site = {r['key']: r['site'] for r in rows}
    tab = [{'class': c, **{s_: sum(1 for r in rows if r['class'] == c and r['site'] == s_) for s_ in sorted(set(site.values()))}}
           for c in CLASSES]
    wcsv(Path(a.out) / 'tables' / 'table_site_by_class.csv', tab); print(tab)
    for sf in sorted((Path(a.out) / 'splits').glob('split_*.json')):
        if '_cv' in sf.stem: continue
        s = json.load(open(sf))
        t = [{'subset': k, **{s_: sum(site[x] == s_ for x in v) for s_ in sorted(set(site.values()))}} for k, v in s.items()]
        wcsv(Path(a.out) / 'tables' / f'table_site_by_subset_{sf.stem}.csv', t); print(sf.stem, t)


def cmd_dupcheck(a):
    """Near-duplicate audit for each split: how many test images have a near-duplicate
    (pHash/embedding cluster) or a same-burst image in train. Uses the audit_per_image.csv
    and split_fruitgroup_per_image.csv produced earlier."""
    aud = {key_of(r.get('path') or r['key']): r for r in csv.DictReader(open(a.audit_csv))}
    grp = {key_of(r.get('path') or r['key']): r['fruit_group'] for r in csv.DictReader(open(a.groups_csv))}
    out = []
    for sf in sorted((Path(a.out) / 'splits').glob('split_*.json')):
        if '_cv' in sf.stem: continue
        s = json.load(open(sf)); tr = set(s['train'])
        dup_tr = {aud[k]['dup_cluster'] for k in tr}; grp_tr = {grp[k] for k in tr}
        te = s['test']
        out.append({'split': sf.stem, 'n_test': len(te),
                    'test_with_near_duplicate_in_train': sum(aud[k]['dup_cluster'] in dup_tr for k in te),
                    'test_sharing_capture_burst_with_train': sum(grp[k] in grp_tr for k in te)})
        days = defaultdict(lambda: defaultdict(int))
        for k_, v in s.items():
            for x in v: days[k_][aud[x]['day']] += 1
        wcsv(Path(a.out) / 'tables' / f'table_day_by_subset_{sf.stem}.csv', [{'subset': k_, **dict(v)} for k_, v in days.items()])
    wcsv(Path(a.out) / 'tables' / 'table_duplicate_audit.csv', out); print(out)


# ============================================================ zero-shot cross-species
def cmd_xspecies(a):
    """Per-image outputs on unlabelled cross-species images (folder/<species>/*.jpg). No accuracy is computed."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rd = first_run(a.out, a.split_stem, seed=a.seed); m, _ = load_ckpt(rd / 'best.pth', dev); rows = []
    for sp in sorted(p for p in Path(a.folder).iterdir() if p.is_dir()):
        for p in sorted(q for q in sp.iterdir() if q.suffix.lower() in EXTS):
            x = tf_eval()(ImageOps.exif_transpose(Image.open(p)).convert('RGB')).unsqueeze(0).to(dev)
            with torch.no_grad():
                o = m(x)
            p5 = F.softmax(o['d'], 1)[0].cpu().numpy(); p2 = F.softmax(o['h'], 1)[0].cpu().numpy()
            fs = per_sample_fss(o['zs'].cpu().numpy(), o['zh'].cpu().numpy())[0]
            rows.append({'species': sp.name, 'file': p.name, 'pred_5class': CLASSES[int(p5.argmax())],
                         'confidence': float(p5.max()), **{f'p_{c}': float(p5[k]) for k, c in enumerate(CLASSES)},
                         'p_diseased': float(p2[0]), 'binary_screen': 'Diseased' if p2[0] >= 0.5 else 'Healthy',
                         'gated_output': CLASSES[int(gate(p5[None], np.array([p2[0]]), 0.5)[0])],
                         'latent_orthogonality_FSS': float(fs), 'ground_truth': 'not available'})
    wcsv(Path(a.out) / 'tables' / 'table_xspecies_per_image.csv', rows)
    summ = []
    for sp in sorted({r['species'] for r in rows}):
        r = [x for x in rows if x['species'] == sp]
        summ.append({'species': sp, 'n': len(r), 'screened_diseased': sum(x['binary_screen'] == 'Diseased' for x in r),
                     **{f'pred_{c}': sum(x['pred_5class'] == c for x in r) for c in CLASSES},
                     'median_confidence': float(np.median([x['confidence'] for x in r])),
                     'FSS_min': float(min(x['latent_orthogonality_FSS'] for x in r)),
                     'FSS_max': float(max(x['latent_orthogonality_FSS'] for x in r)), 'ground_truth': 'not available'})
    wcsv(Path(a.out) / 'tables' / 'table_xspecies_summary.csv', summ); print(summ)


# ============================================================ Fig. 2 schematic
def cmd_fig2(a):
    plt = plot_style()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 6.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis('off')
    def box(x, y, w, h, txt, col, fs=13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=col, ec='black', lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs)
    def arr(x1, y1, x2, y2, ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=1.4, ls=ls, color='black'))
    c1, c2, cg = '#CFE8F3', '#FBE3C4', '#E6E6E6'
    ax.text(2.9, 5.95, 'Agronomist diagnostic process', ha='center', fontsize=15, weight='bold')
    ax.text(9.6, 5.95, 'DSDN', ha='center', fontsize=15, weight='bold')
    box(0.4, 3.75, 5.0, 1.7, 'Stage 1: Health screening\nHealthy or diseased?\n(colour uniformity, intact skin,\nabsence of lesions)', c1)
    box(0.4, 0.75, 5.0, 2.1, 'Stage 2: Disease typing\nAlternaria / Anthracnose /\nBacterial blight / Cercospora\n(lesion colour, margin, texture)', c2)
    arr(2.9, 3.75, 2.9, 2.85); ax.text(3.0, 3.25, 'if diseased', fontsize=12, style='italic')
    box(6.9, 4.85, 5.5, 0.8, 'Shared EfficientNet-B0 backbone → $f_{enc}$', cg)
    box(6.6, 3.4, 2.7, 0.95, 'Healthy head → $z_h$', c1); box(10.0, 3.4, 2.7, 0.95, 'Symptom head → $z_s$', c2)
    arr(9.0, 4.85, 7.95, 4.35); arr(10.2, 4.85, 11.35, 4.35)
    box(6.6, 1.9, 2.7, 0.95, 'Binary output\nhealthy / diseased', c1); box(10.0, 1.9, 2.7, 0.95, 'Five-class output\ndisease type', c2)
    arr(7.95, 3.4, 7.95, 2.85); arr(11.35, 3.4, 11.35, 2.85)
    ax.annotate('', xy=(10.0, 3.87), xytext=(9.3, 3.87), arrowprops=dict(arrowstyle='<->', lw=1.2, ls='--'))
    ax.text(9.65, 4.05, '$\\perp$', ha='center', fontsize=14)
    box(6.6, 0.35, 6.1, 0.95, 'Two-stage output: Healthy if P(diseased) < τ,\notherwise disease type from five-class head', '#FFFFFF', 12)
    arr(7.95, 1.9, 7.95, 1.3); arr(11.35, 1.9, 11.35, 1.3)
    ax.text(3.0, 0.25, 'Matching colours: stage ↔ projection head', ha='center', fontsize=12, style='italic')
    for ext in ['png', 'pdf']:
        fig.savefig(Path(a.out) / f'fig2_two_stage.{ext}')
    print('saved fig2_two_stage.png/.pdf')


# ============================================================ summary tables
def cmd_summarize(a):
    from scipy.stats import ttest_rel
    runs = defaultdict(dict)
    for mf in (Path(a.out) / 'runs').glob('*/metrics.json'):
        sp, cfg, sd = mf.parent.name.split('__'); runs[(sp, cfg)][sd] = json.load(open(mf))
    tdir = Path(a.out) / 'tables'; tdir.mkdir(exist_ok=True)
    ms = lambda v: f'{np.mean(v) * 100:.2f} ± {np.std(v, ddof=1) * 100:.2f}' if len(v) > 1 else f'{np.mean(v) * 100:.2f}'
    msr = lambda v: f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if len(v) > 1 else f'{np.mean(v):.4f}'
    # ablation and split comparison (test set)
    rows = []
    for (sp, cfg), d in sorted(runs.items()):
        if '_cv' in sp: continue
        T = [v['test'] for v in d.values()]
        r = {'split': sp, 'config': cfg, 'n_seeds': len(T), 'params_M': round(list(d.values())[0]['params'] / 1e6, 2),
             'acc_%': ms([t['acc'] for t in T]), 'f1_weighted': msr([t['f1_weighted'] for t in T]),
             'mcc': msr([t['mcc'] for t in T]),
             'fss': msr([t['fss'] for t in T]) if 'fss' in T[0] else 'n/a (single head)'}
        full = runs.get((sp, 'full'))
        if full and cfg != 'full' and 's42' in full and 's42' in d:
            r['mcnemar_p_vs_full_seed42'] = mcnemar_p(Path(a.out) / 'runs' / f'{sp}__full__s42',
                                                       Path(a.out) / 'runs' / f'{sp}__{cfg}__s42')
        if full and cfg != 'full':
            common = sorted(set(full) & set(d))
            if len(common) >= 2:
                r['paired_t_p_vs_full_acc'] = float(ttest_rel([full[s]['test']['acc'] for s in common], [d[s]['test']['acc'] for s in common]).pvalue)
        rows.append(r)
    wcsv(tdir / 'table_ablation_and_splits.csv', rows)
    # gated vs direct inference
    r2 = []
    for (sp, cfg), d in sorted(runs.items()):
        if '_cv' in sp or 'screening' not in list(d.values())[0]: continue
        for variant in ['direct_5class_head', 'gated_tau0.5', 'gated_tau_val', 'direct_5class_plain']:
            V = [v['screening'][variant] for v in d.values() if variant in v['screening']]
            if V:
                r2.append({'split': sp, 'model': cfg, 'inference': variant,
                           **{k: ms([x[k] for x in V]) for k in V[0]},
                           'tau_val': '' if 'gated_tau_val' != variant else ','.join(str(v['screening']['tau_selected_on_val']) for v in d.values())})
    wcsv(tdir / 'table_screening_gated_vs_direct.csv', r2)
    # Class-wise F1 and FSS (Tables 9, 12)
    cw = []
    for (sp, cfg), d in sorted(runs.items()):
        if cfg != 'full' or '_cv' in sp: continue
        T = [v['test'] for v in d.values()]
        for c in CLASSES:
            cw.append({'split': sp, 'class': c, 'f1': msr([t[f'f1_{c}'] for t in T]), 'fss': msr([t[f'fss_{c}'] for t in T])})
    wcsv(tdir / 'table_classwise_f1_fss.csv', cw)
    # Cross-validation (test fixed, 5 grouped folds)
    cv = defaultdict(list)
    for (sp, cfg), d in runs.items():
        if '_cv' in sp:
            for v in d.values(): cv[(sp.split('_cv')[0], cfg)].append(v)
    wcsv(tdir / 'table_cv.csv', [{'split': k[0], 'config': k[1], 'n_folds': len(v),
                                   'val_acc_%': ms([x['val']['acc'] for x in v]), 'test_acc_%': ms([x['test']['acc'] for x in v])}
                                  for k, v in cv.items()])
    # latent probing
    pr = defaultdict(list)
    for pf in (Path(a.out) / 'runs').glob('*/probe.json'):
        sp, cfg, _ = pf.parent.name.split('__'); pr[(sp, cfg)].append(json.load(open(pf)))
    wcsv(tdir / 'table_latent_probing.csv', [{'split': k[0], 'config': k[1], 'n_seeds': len(v),
                                          **{c: msr([x[c] for x in v]) for c in v[0] if '->' in c}} for k, v in pr.items()])
    print('tables written to', tdir)
    for f in sorted(tdir.glob('*.csv')): print(' ', f.name)


def mcnemar_p(rd_a, rd_b):
    """Exact McNemar test on per-image test correctness of two runs (same split)."""
    from scipy.stats import binomtest
    A = {r['key']: int(r['y']) == int(r['pred_5class']) for r in csv.DictReader(open(Path(rd_a) / 'predictions_test.csv'))}
    B = {r['key']: int(r['y']) == int(r['pred_5class']) for r in csv.DictReader(open(Path(rd_b) / 'predictions_test.csv'))}
    b = sum(A[k] and not B[k] for k in A); c = sum(B[k] and not A[k] for k in A)
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0




# ============================================================ manuscript tables 7-10 and figures 4-8
def _runs(out, split_stem, cfg):
    return sorted(p.parent for p in (Path(out) / 'runs').glob(f'{split_stem}__{cfg}__s*/metrics.json'))


def _pred(rd):
    r = list(csv.DictReader(open(Path(rd) / 'predictions_test.csv')))
    r.sort(key=lambda x: x['key'])
    return [x['key'] for x in r], np.array([int(x['y']) for x in r]), np.array([int(x['pred_5class']) for x in r]), r


def _boot_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def cmd_stats(a):
    """Table 8 (accuracy, F1, 5-fold CV, bootstrap 95 % CI, exact McNemar vs DSDN with Bonferroni),
    Table 9 (class-wise F1), Table 10 (sensitivity, validation set). Seed 42 = reference run."""
    from sklearn.metrics import f1_score
    sp, out = a.split_stem, Path(a.out); tdir = out / 'tables'; tdir.mkdir(exist_ok=True)
    acc = lambda y, p: float((y == p).mean())
    f1w = lambda y, p: float(f1_score(y, p, average='weighted'))
    ref = out / 'runs' / f'{sp}__full__s{a.seed}'
    kref, yref, pref, _ = _pred(ref)
    models = ['full'] + [b for b in BASELINES if _runs(out, sp, b)]
    n_cmp = len(models) - 1
    t8, t9 = [], []
    for cfg in models:
        rds = _runs(out, sp, cfg)
        M = [json.load(open(r / 'metrics.json'))['test'] for r in rds]
        r42 = out / 'runs' / f'{sp}__{cfg}__s{a.seed}'
        rr = r42 if (r42 / 'metrics.json').exists() else rds[0]
        k, y, p, _ = _pred(rr); assert k == kref, 'test sets differ'
        A = [m['acc'] for m in M]; Fw = [m['f1_weighted'] for m in M]
        sd = lambda v, sc=100: f' ± {np.std(v, ddof=1) * sc:.2f}' if len(v) > 1 else ''
        cv = [json.load(open(r / 'metrics.json'))['test']['acc'] for r in (out / 'runs').glob(f'{sp}_cv*__{cfg}__s*')
              if (r / 'metrics.json').exists()]
        lo, hi = _boot_ci(y, p, acc)
        row = {'Model': BL_NAMES.get(cfg, cfg), 'n_seeds': len(M),
               'Acc (%)': f'{np.mean(A) * 100:.2f}{sd(A)}', 'F1 (weighted)': f'{np.mean(Fw):.4f}{sd(Fw, 1)}',
               '5-fold CV acc (%)': (f'{np.mean(cv) * 100:.2f} ± {np.std(cv, ddof=1) * 100:.2f}' if len(cv) > 1 else '—'),
               f'95% CI acc, seed {a.seed} (%)': f'[{lo * 100:.2f}, {hi * 100:.2f}]',
               'Params (M)': round(n_params(build(cfg, pretrained=False)) / 1e6, 2)}
        if cfg != 'full':
            b = int(((pref == yref) & (p != y)).sum()); c = int(((p == y) & (pref != yref)).sum())
            from scipy.stats import binomtest
            pv = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
            row.update({'DSDN right / model wrong': b, 'model right / DSDN wrong': c,
                        'McNemar p': pv, f'Bonferroni p (x{n_cmp})': min(1.0, pv * n_cmp),
                        'Delta acc vs DSDN (pp)': round(float(np.mean(A) - json.load(open(ref / 'metrics.json'))['test']['acc']) * 100, 2)})
        t8.append(row)
        t9.append({'Model': BL_NAMES.get(cfg, cfg), **{c_: round(float(f1_score(y == i, p == i)), 3) for i, c_ in enumerate(CLASSES)},
                   'Macro F1': round(float(f1_score(y, p, average='macro')), 3)})
    wcsv(tdir / 'table8_comparison.csv', t8); wcsv(tdir / 'table9_classwise_f1.csv', t9)
    print('\n'.join(str(r) for r in t8))
    # Table 10 sensitivity (validation set, as in the manuscript)
    t10 = []
    for cfg, (pn, v) in [('full', ('-', '-'))] + list(SENS.items()):
        rd = out / 'runs' / f'{sp}__{cfg}__s{a.seed}'
        if not (rd / 'metrics.json').exists(): continue
        m = json.load(open(rd / 'metrics.json'))
        t10.append({'config': cfg, 'coefficient': {'a': 'alpha', 'b': 'beta', 'g': 'gamma'}.get(pn, 'selected'),
                    'value': v if cfg != 'full' else 'alpha=0.5, beta=0.1, gamma=0.2',
                    'val_f1_weighted': round(m['val']['f1_weighted'], 4), 'val_fss': round(m['val']['fss'], 4),
                    'test_acc': round(m['test']['acc'], 4)})
    if len(t10) > 1:
        wcsv(tdir / 'table10_sensitivity.csv', t10); print('\n'.join(str(r) for r in t10))


def cmd_benchmark(a):
    """Table 7: parameters and single-image GPU latency (batch 1, 224 px, FP32, 50 warm-up + N timed runs)."""
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for cfg in ['full'] + list(BASELINES):
        m = build(cfg, pretrained=False).to(dev).eval(); x = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            for _ in range(50): m(x)
            if dev == 'cuda': torch.cuda.synchronize()
            ts = []
            for _ in range(a.bench_runs):
                t0 = time.perf_counter(); m(x)
                if dev == 'cuda': torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        rows.append({'Model': BL_NAMES.get(cfg, cfg), 'Params (M)': round(n_params(m) / 1e6, 2),
                     'Latency mean (ms)': round(float(np.mean(ts)), 2), 'Latency SD (ms)': round(float(np.std(ts)), 2),
                     'device': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu', 'runs': a.bench_runs})
        print(rows[-1]); del m
        if dev == 'cuda': torch.cuda.empty_cache()
    wcsv(Path(a.out) / 'tables' / 'table7_complexity.csv', rows)


def _panel(plt, fig, name, out):
    d = Path(out) / 'figures'; d.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(d / f'{name}.{ext}')
    plt.close(fig)


def _stack(plt, names, out, final):
    """Stacked vertical version of separately saved panels, each labelled (a), (b), ..."""
    d = Path(out) / 'figures'; names = [n for n in names if (d / f'{n}.png').exists()]
    if not names: return
    ims = [plt.imread(d / f'{n}.png') for n in names]
    fig, ax = plt.subplots(len(ims), 1, figsize=(8, 6 * len(ims)))
    for i, (x, im) in enumerate(zip(np.atleast_1d(ax), ims)):
        x.imshow(im); x.axis('off'); x.set_title(f'({chr(97 + i)})', loc='left', fontsize=16)
    _panel(plt, fig, final, out)


def cmd_figures(a):
    plt = plot_style(); out = Path(a.out); sp = a.split_stem
    from sklearn.metrics import confusion_matrix
    ref = out / 'runs' / f'{sp}__full__s{a.seed}'
    # ---- Fig. 4: five-class confusion matrix (a) and binary screening confusion matrix (b)
    _, y, p, rows = _pred(ref)
    for name, cm, labels in [('fig4a_cm_5class', confusion_matrix(y, p, labels=range(5)), SHORT),
                             ('fig4b_cm_binary', confusion_matrix((y == HEALTHY).astype(int),
                                                                  np.array([int(r['pred_binary']) for r in rows]), labels=[0, 1]),
                              ['Diseased', 'Healthy'])]:
        fig, ax = plt.subplots(figsize=(6.5, 5.5)); ax.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for s_ in ax.spines.values(): s_.set_visible(False)
        _panel(plt, fig, name, out)
    _stack(plt, ['fig4a_cm_5class', 'fig4b_cm_binary'], out, 'fig4_stacked')
    # ---- Fig. 5: training curves, mean over seeds (shaded = min-max)
    logs = [list(csv.DictReader(open(r / 'epoch_log.csv'))) for r in _runs(out, sp, 'full')]
    E = min(len(l) for l in logs); ep = np.arange(1, E + 1)
    G = lambda k: np.array([[float(l[i][k]) for i in range(E)] for l in logs])
    best = int(json.load(open(ref / 'metrics.json'))['best_epoch'])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k, lab, col in [('train_acc', 'Training', CB[0]), ('val_acc', 'Validation', CB[1])]:
        v = G(k) * 100; ax.plot(ep, v.mean(0), color=col, lw=2, label=lab); ax.fill_between(ep, v.min(0), v.max(0), color=col, alpha=0.2)
    ax.axvline(best, ls='--', color='k', lw=1); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.legend(frameon=False)
    from matplotlib.ticker import MaxNLocator; ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _panel(plt, fig, 'fig5a_accuracy', out)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (k, lab) in enumerate([('loss_total', '$L_{total}$'), ('loss_cls', '$L_{cls}$'), ('loss_bin', '$L_{bin}$'),
                                  ('loss_recon', '$L_{recon}$'), ('loss_ortho', '$L_{ortho}$')]):
        v = G(k); ax.plot(ep, v.mean(0), color=CB[i], lw=2, label=lab); ax.fill_between(ep, v.min(0), v.max(0), color=CB[i], alpha=0.15)
    ax.set_yscale('log'); ax.axvline(best, ls='--', color='k', lw=1); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, ncol=2, fontsize=12); _panel(plt, fig, 'fig5b_loss', out)
    _stack(plt, ['fig5a_accuracy', 'fig5b_loss'], out, 'fig5_stacked')
    # ---- Fig. 6: sensitivity bars
    t10 = Path(out) / 'tables' / 'table10_sensitivity.csv'
    if t10.exists():
        T = list(csv.DictReader(open(t10))); full_r = [r for r in T if r['config'] == 'full'][0]; names = []
        for co, sym, vals in [('alpha', 'α', ['0.3', '0.5', '0.7']), ('beta', 'β', ['0.05', '0.1', '0.2']), ('gamma', 'γ', ['0.1', '0.2', '0.3'])]:
            R = {r['value']: r for r in T if r['coefficient'] == co}; sel = {'alpha': '0.5', 'beta': '0.1', 'gamma': '0.2'}[co]
            R[sel] = full_r; vs = [v for v in vals if v in R]
            if len(vs) < 2: continue
            fig, ax = plt.subplots(figsize=(6, 4.5)); x = np.arange(len(vs)); w = 0.38
            ax.bar(x - w / 2, [float(R[v]['val_f1_weighted']) for v in vs], w, color=CB[0], label='Val F1')
            ax.bar(x + w / 2, [float(R[v]['val_fss']) for v in vs], w, color=CB[1], label='FSS')
            ax.set_xticks(x); ax.set_xticklabels([f'{sym} = {v}' + (' (sel.)' if v == sel else '') for v in vs])
            lo = min(min(float(R[v]['val_f1_weighted']), float(R[v]['val_fss'])) for v in vs)
            ax.set_ylim(max(0, lo - 0.02), 1.0); ax.legend(frameon=False, fontsize=12); _panel(plt, fig, f'fig6_{co}', out); names.append(f'fig6_{co}')
        if names: _stack(plt, names, out, 'fig6_stacked')
    # ---- Fig. 7: weighted F1 (seed-42 test set) with bootstrap 95 % CI, all models
    from sklearn.metrics import f1_score
    models = ['full'] + [b for b in BASELINES if (out / 'runs' / f'{sp}__{b}__s{a.seed}' / 'metrics.json').exists()]
    if len(models) > 1:
        V = []
        for cfg in models:
            _, y_, p_, _ = _pred(out / 'runs' / f'{sp}__{cfg}__s{a.seed}')
            f = f1_score(y_, p_, average='weighted'); lo, hi = _boot_ci(y_, p_, lambda u, v: f1_score(u, v, average='weighted'))
            V.append((BL_NAMES[cfg], f, lo, hi))
        V.sort(key=lambda t: t[1])
        fig, ax = plt.subplots(figsize=(8, 6)); yy = np.arange(len(V))
        ax.barh(yy, [v[1] for v in V], color=[CB[5] if v[0] == 'DSDN' else CB[0] for v in V],
                xerr=[[v[1] - v[2] for v in V], [v[3] - v[1] for v in V]], capsize=3)
        ax.set_yticks(yy); ax.set_yticklabels([v[0] for v in V]); ax.set_xlabel('Weighted F1 (test set)')
        ax.set_xlim(max(0, min(v[2] for v in V) - 0.02), 1.0); _panel(plt, fig, 'fig7_f1_comparison', out)
    # ---- Fig. 8: t-SNE of z_h (a) and z_s (b); Fig. S3: PCA
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    P = np.load(ref / 'pred_test.npz'); yy = P['y']
    for z, lab in [('zh', 'healthy'), ('zs', 'symptom')]:
        for meth, fn in [('tsne', lambda X: TSNE(2, random_state=0, init='pca', perplexity=30).fit_transform(X)),
                         ('pca', lambda X: PCA(2).fit_transform(X))]:
            if meth == 'tsne' and len(P[z]) <= 31: continue
            T = fn(P[z]); fig, ax = plt.subplots(figsize=(6.5, 6))
            for i in range(5):
                ax.scatter(T[yy == i, 0], T[yy == i, 1], s=10, color=CB[i], label=SHORT[i])
            ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, markerscale=2, fontsize=12)
            _panel(plt, fig, f'fig8_{meth}_{lab}_head', out)
    _stack(plt, ['fig8_tsne_healthy_head', 'fig8_tsne_symptom_head'], out, 'fig8_stacked')
    _stack(plt, ['fig8_pca_healthy_head', 'fig8_pca_symptom_head'], out, 'figS3_pca_stacked')
    print('figures written to', out / 'figures')


# ============================================================ PILOT (run first)
def cmd_pilot(a):
    """End-to-end check on a small real subset (about 15-20 min on a T4). Writes out/pilot/pilot_report.txt.
    Checks: architecture, parameter count, label convention, every config trains, losses finite and falling,
    FSS rises, evaluation + gated inference, Grad-CAM++ maps, kill-and-resume, Drive writes, time estimate."""
    import subprocess, sys
    from types import SimpleNamespace
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    po = Path(a.out) / 'pilot'; (po / 'splits').mkdir(parents=True, exist_ok=True)
    rep = []; ok_all = True
    def check(name, cond, info=''):
        nonlocal ok_all
        ok_all &= bool(cond); line = f'[{"PASS" if cond else "FAIL"}] {name}  {info}'; rep.append(line); print(line)
    check('GPU available', dev == 'cuda', torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU only')
    # --- cache present
    n_cache = sum(1 for q in Path(a.cache).rglob('*.jpg') if not q.name.endswith('.tmp.jpg'))
    check('local cache complete', n_cache >= 5099 or n_cache == len(list_images(a.data)), f'{n_cache} images')
    # --- pilot split from the real split (train/val/test kept apart)
    full = json.load(open(a.split)); rng = random.Random(0); ps = {}
    for sub, n in [('train', 60), ('val', 15), ('test', 15)]:
        ps[sub] = []
        for i in range(5):
            ks = [k for k in full[sub] if label_of_key(k) == i]; ps[sub] += rng.sample(ks, min(n, len(ks)))
    sp = po / 'splits' / 'pilot.json'; json.dump(ps, open(sp, 'w'))
    check('pilot split has no overlap', not (set(ps['train']) & set(ps['test'])) and not (set(ps['train']) & set(ps['val'])),
          f"train {len(ps['train'])} / val {len(ps['val'])} / test {len(ps['test'])}")
    check('main split has no overlap', not (set(full['train']) & set(full['test'])) and not (set(full['val']) & set(full['test'])),
          f"{Path(a.split).name}: {len(full['train'])}/{len(full['val'])}/{len(full['test'])}")
    # --- architecture
    m = build('full', pretrained=True).to(dev).eval()
    keys = sorted({k.split('.')[0] for k in m.state_dict()})
    check('modules', keys == ['decoder', 'disease_classifier', 'encoder', 'healthy_classifier', 'healthy_encoder', 'symptom_encoder'], keys)
    check('parameter count (paper must state this value)', abs(n_params(m) - 5_530_211) < 1000, f'{n_params(m):,}')
    with torch.no_grad():
        o = m(torch.randn(2, 3, HP['img'], HP['img'], device=dev))
    shp = {k: tuple(v.shape) for k, v in o.items() if v is not None}
    check('output shapes', shp == {'d': (2, 5), 'h': (2, 2), 'zs': (2, 128), 'zh': (2, 128), 'rec': (2, 1280), 'f': (2, 1280)}, shp)
    for cfg in CONFIGS:
        mm = build(cfg, pretrained=False); rep.append(f'       {cfg:15s} params {n_params(mm):,}')
    # --- label convention
    ds = KeyDS(ps['train'][:5] + [k for k in ps['train'] if label_of_key(k) == HEALTHY][:1], a.cache, tf_eval())
    hb = [ds[i][2] for i in range(len(ds))]; yy = [ds[i][1] for i in range(len(ds))]
    check('binary label: 1 = healthy, 0 = diseased', all((h == 1) == (y == HEALTHY) for h, y in zip(hb, yy)))
    # --- every config trains 1 epoch; full trains 4 epochs
    for cfg in CONFIGS:
        ep = 4 if cfg == 'full' else 1
        args = SimpleNamespace(out=str(po), split=str(sp), cache=a.cache, config=cfg, seed=1, epochs=ep,
                               no_pretrained=False, force=True, keep_last=False, patience=0)
        rd = po / 'runs' / run_name(args)
        if rd.exists(): shutil.rmtree(rd)
        t0 = time.time()
        try:
            cmd_train(args); err = ''
        except Exception as e:
            err = repr(e)[:200]
        mj = rd / 'metrics.json'
        check(f'train+evaluate {cfg}', mj.exists() and not err, err or f'{time.time() - t0:.0f}s')
    log = list(csv.DictReader(open(po / 'runs' / 'pilot__full__s1' / 'epoch_log.csv')))
    L = [float(r['loss_total']) for r in log]; Lo = [float(r['loss_ortho']) for r in log]; fs = [float(r['val_fss']) for r in log]
    check('losses finite', all(math.isfinite(x) for r in log for k, x in r.items() if k.startswith('loss') for x in [float(x)]))
    check('total loss decreases', L[-1] < L[0], f'{L[0]:.3f} -> {L[-1]:.3f}')
    check('orthogonality loss decreases', Lo[-1] < Lo[0], f'{Lo[0]:.4f} -> {Lo[-1]:.4f}')
    check('FSS increases', fs[-1] > fs[0], f'{fs[0]:.3f} -> {fs[-1]:.3f}')
    mt = json.load(open(po / 'runs' / 'pilot__full__s1' / 'metrics.json'))
    check('5-class learns (test acc > 0.5 after 4 epochs on 300 images)', mt['test']['acc'] > 0.5, f"{mt['test']['acc']:.3f}")
    check('binary head learns (test acc > 0.7)', mt['test']['bin_acc'] > 0.7, f"{mt['test']['bin_acc']:.3f}")
    check('gated vs direct computed', 'screening' in mt and 'gated_tau0.5' in mt['screening'], f"tau_val={mt.get('screening', {}).get('tau_selected_on_val')}")
    pr = list(csv.DictReader(open(po / 'runs' / 'pilot__full__s1' / 'predictions_test.csv')))
    check('per-image predictions saved', len(pr) == len(ps['test']) and 'p_diseased' in pr[0], f'{len(pr)} rows')
    # --- Grad-CAM++ both maps
    mm, _ = load_ckpt(po / 'runs' / 'pilot__full__s1' / 'best.pth', dev)
    x = tf_eval()(Image.open(Path(a.cache) / ps['test'][0]).convert('RGB')).unsqueeze(0).to(dev)
    cs, _ = gradcampp(mm, x, 'symptom'); ch, _ = gradcampp(mm, x, 'healthy')
    check('Grad-CAM++ maps valid', np.isfinite(cs).all() and np.isfinite(ch).all() and cs.max() > 0 and ch.max() > 0,
          f'SCS symptom {scs(cs):.3f}, healthy {scs(ch):.3f}')
    # --- kill-and-resume in a separate process
    rd = po / 'runs' / 'pilot__full__s2'
    if rd.exists(): shutil.rmtree(rd)
    cmd = [sys.executable, __file__, 'train', '--cache', a.cache, '--out', str(po), '--split', str(sp),
           '--config', 'full', '--seed', '2', '--epochs', '3']
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while p.poll() is None and not (rd / 'last.pth').exists(): time.sleep(1)
    time.sleep(2); p.kill(); p.wait()
    killed_at = len(list(csv.DictReader(open(rd / 'epoch_log.csv')))) if (rd / 'epoch_log.csv').exists() else 0
    res = subprocess.run(cmd, capture_output=True, text=True); out = res.stdout
    ok = 'RESUMING' in out and (rd / 'metrics.json').exists()
    check('kill-and-resume', ok, f'killed after epoch {killed_at}, ' + ('resumed and finished' if ok else
          'resume FAILED: ' + (res.stderr.strip().splitlines() or ['no error text'])[-1][:300]))
    if ok:
        lg = list(csv.DictReader(open(rd / 'epoch_log.csv')))
        check('resumed run has all epochs once', [int(r['epoch']) for r in lg] == list(range(1, len(lg) + 1)) and len(lg) == 3,
              f'epochs logged: {[int(r["epoch"]) for r in lg]}')
    # --- time estimate on the real training set
    ld = make_loader(full['train'], a.cache, True, gen_seed=0); m = build('full').to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3); scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    t0 = None; nb = 0
    for i, (x, y, hb, _) in enumerate(ld):
        if i == 5: t0 = time.time()
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(device_type='cuda', enabled=dev == 'cuda'):
            loss = F.cross_entropy(m(x)['d'], y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if i >= 5: nb += 1
        if nb == 60: break
    sec_b = (time.time() - t0) / nb
    ep_s = sec_b * (len(full['train']) / HP['bs']) * 1.25   # +25 % for validation
    run_min = ep_s * HP['epochs'] / 60
    rep.append(f'       time: {sec_b:.3f} s/batch -> about {ep_s:.0f} s/epoch -> about {run_min:.0f} min per 25-epoch run'
               f' -> 17 runs about {17 * run_min / 60:.1f} h, 35 runs about {35 * run_min / 60:.1f} h')
    print(rep[-1])
    # --- Drive write check
    try:
        t = Path(a.out) / '.write_test'; t.write_text('ok'); t.unlink(); check('output folder writable', True, str(Path(a.out)))
    except Exception as e:
        check('output folder writable', False, repr(e))
    rep.append('ALL CHECKS PASSED - start the full runs' if ok_all else 'SOME CHECKS FAILED - send pilot_report.txt')
    (po / 'pilot_report.txt').write_text('\n'.join(rep)); print('\n' + rep[-1]); print('report:', po / 'pilot_report.txt')

# ============================================================ CLI
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--data'); ap.add_argument('--cache'); ap.add_argument('--out')
    ap.add_argument('--split'); ap.add_argument('--split_stem', default='split_random')
    ap.add_argument('--config', default='full'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=0); ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--groups_json'); ap.add_argument('--groups_csv'); ap.add_argument('--audit_csv')
    ap.add_argument('--annot'); ap.add_argument('--n', type=int, default=15); ap.add_argument('--n_wrong', type=int, default=6)
    ap.add_argument('--folder'); ap.add_argument('--site_km', type=float, default=1.0)
    ap.add_argument('--bench_runs', type=int, default=500)
    ap.add_argument('--patience', type=int, default=0, help='early-stopping patience (default 10, as DSDN)')
    ap.add_argument('--cache_drive', help='Drive folder for the persistent resized cache')
    ap.add_argument('--force', action='store_true', help='redo even if results exist')
    ap.add_argument('--keep_last', action='store_true', help='keep last.pth after a run finishes')
    a = ap.parse_args()
    {'cache': cmd_cache, 'split_random': cmd_split_random, 'split_grouped': cmd_split_grouped, 'cv_splits': cmd_cv_splits,
     'train': cmd_train, 'probe': cmd_probe, 'scs': cmd_scs, 'fig9': cmd_fig9, 'export_annot': cmd_export_annot,
     'point': cmd_point, 'farms': cmd_farms, 'dupcheck': cmd_dupcheck, 'xspecies': cmd_xspecies,
     'fig2': cmd_fig2, 'summarize': cmd_summarize, 'pilot': cmd_pilot,
     'stats': cmd_stats, 'benchmark': cmd_benchmark, 'figures': cmd_figures}[a.cmd](a)
