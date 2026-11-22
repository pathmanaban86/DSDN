# %% [markdown]
# # Pomegranate dataset audit (run BEFORE any model training)
# Answers five questions that decide the whole study design:
#   Q1  What is in the dataset? (counts, sizes, formats, corrupt files)
#   Q2  Are there exact / near duplicates, and how big are the clusters?
#   Q3  Can images be grouped by acquisition session (EXIF time, burned-in timestamp)?
#   Q4  Is there a shortcut? (class predictable from background / colour / size alone)
#   Q5  How much does a random split inflate accuracy vs a grouped split?
# Outputs go to OUT_DIR: CSV tables, summary.json, and figures (one file per panel).
#
# Colab setup:
#   !pip install -q timm imagehash easyocr

# %% ---------------------------------------------------------------- config
from pathlib import Path
DATA_DIR = Path('data/Pomegranate Diseases Dataset')   # folder with the five class sub-folders
OUT_DIR = Path('audit')
CLASSES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']
SEED = 42
RUN_OCR = True            # read the burned-in date/time stamp (bottom-right corner) if EXIF is missing
SESSION_GAP_MIN = 10      # images closer than this in time (same class folder) = same capture session
PHASH_THR = 8             # Hamming distance (64-bit pHash) for "near duplicate"
EMB_THR = 0.95            # cosine similarity (ImageNet embeddings) for "near duplicate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------- imports & plot style
import json, hashlib, re, warnings
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np, pandas as pd
from PIL import Image, ExifTags, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

CB = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']  # Okabe-Ito
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 14, 'axes.grid': False, 'axes.facecolor': 'white',
                     'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})


def savefig(fig, name):
    fig.savefig(OUT_DIR / f'{name}.png'); fig.savefig(OUT_DIR / f'{name}.pdf'); plt.close(fig)


SUMMARY = {}

# %% ---------------------------------------------------------------- Q1 inventory
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def exif_info(im):
    try:
        ex = im._getexif() or {}
    except Exception:
        ex = {}
    g = lambda n: ex.get(TAGS.get(n))
    return {'exif_time': g('DateTimeOriginal') or g('DateTime'), 'make': g('Make'), 'model': g('Model'),
            'gps': TAGS.get('GPSInfo') in ex}


rows = []
for ci, c in enumerate(CLASSES):
    d = DATA_DIR / c
    if not d.exists():
        d = next(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.lower() == c.lower())
    for p in sorted(d.rglob('*')):
        if p.suffix.lower() not in EXTS:
            continue
        r = {'path': str(p), 'file': p.name, 'class': c, 'label': ci, 'bytes': p.stat().st_size}
        try:
            with Image.open(p) as im:
                r.update(width=im.size[0], height=im.size[1], mode=im.mode, fmt=im.format, **exif_info(im))
                im.verify()
            r['corrupt'] = False
        except Exception as e:
            r['corrupt'] = True; r['error'] = str(e)[:80]
        rows.append(r)
df = pd.DataFrame(rows)
df['md5'] = [hashlib.md5(open(p, 'rb').read()).hexdigest() for p in df.path]
print(df.groupby('class').size())
SUMMARY['Q1'] = {
    'n_images': int(len(df)), 'per_class': df['class'].value_counts().to_dict(),
    'corrupt': int(df.corrupt.sum()),
    'resolutions_top5': {f'{w}x{h}': int(n) for (w, h), n in Counter(zip(df.width, df.height)).most_common(5)},
    'formats': df.fmt.value_counts().to_dict(),
    'exif_time_present': int(df.exif_time.notna().sum()),
    'camera_models': df.model.value_counts(dropna=False).head(10).astype(int).to_dict(),
    'gps_present': int(df.gps.fillna(False).sum()),
    'filename_patterns': df.file.str.replace(r'\d+', '#', regex=True).value_counts().head(10).to_dict(),
}
# Resolution/size by class: if one class differs in resolution or file size, a model can cheat on it.
size_tab = df.groupby('class').agg(w_med=('width', 'median'), h_med=('height', 'median'),
                                   kb_med=('bytes', lambda b: np.median(b) / 1024))
size_tab.to_csv(OUT_DIR / 'q1_size_by_class.csv'); print(size_tab)

# %% ---------------------------------------------------------------- Q3a timestamps (EXIF, then OCR of burned-in stamp)
def parse_time(s):
    if not isinstance(s, str):
        return None
    s = s.strip().replace('/', ':').replace('-', ':')
    for fmt in ('%Y:%m:%d %H:%M:%S', '%Y:%m:%d %H:%M', '%Y:%m:%d'):
        try:
            return datetime.strptime(s[:len(datetime.now().strftime(fmt))], fmt)
        except Exception:
            pass
    m = re.search(r'(20\d{2})\D{0,2}(\d{1,2})\D{0,2}(\d{1,2})\D{1,3}(\d{1,2})\D(\d{2})', s)
    if m:
        y, mo, d, h, mi = map(int, m.groups())
        try:
            return datetime(y, mo, d, h, mi)
        except ValueError:
            return None
    return None


df['time'] = df.exif_time.map(parse_time)
need = df.time.isna()
if RUN_OCR and need.any():
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    ocr_txt = []
    for p in df.path[need]:
        im = ImageOps.exif_transpose(Image.open(p).convert('RGB'))
        w, h = im.size
        crop = im.crop((int(w * 0.55), int(h * 0.90), w, h)).resize((900, int(900 * 0.1 * h / (0.45 * w) + 1)))
        txt = ' '.join(reader.readtext(np.asarray(crop), detail=0))
        ocr_txt.append(txt)
    df.loc[need, 'ocr_text'] = ocr_txt
    df.loc[need, 'time'] = df.loc[need, 'ocr_text'].map(parse_time)
SUMMARY['Q3_time_coverage'] = {'with_time': int(df.time.notna().sum()), 'of': int(len(df)),
                               'source': 'EXIF' if df.exif_time.notna().any() else 'OCR'}

# Sessions: sort each class by time; a gap > SESSION_GAP_MIN starts a new session.
# NOTE: sessions are formed across all classes too ('day' groups) because one farm visit
# usually covers several diseases.
df['day'] = df.time.map(lambda t: t.strftime('%Y-%m-%d') if t else 'unknown')
sess = []
for c, g in df.sort_values('time').groupby('class'):
    k, last = 0, None
    for i, t in zip(g.index, g.time):
        if t is None or pd.isna(t):
            sess.append((i, f'{c}_unk_{i}')); continue
        if last is not None and (t - last).total_seconds() / 60 > SESSION_GAP_MIN:
            k += 1
        sess.append((i, f'{c}_s{k:03d}')); last = t
df['session'] = pd.Series(dict(sess))
st = df[df.time.notna()].groupby('class').agg(n_days=('day', 'nunique'), n_sessions=('session', 'nunique'),
                                               first=('time', 'min'), last=('time', 'max'))
st['imgs_per_session'] = df[df.time.notna()].groupby('class').size() / st.n_sessions
st.to_csv(OUT_DIR / 'q3_sessions_by_class.csv'); print(st)
SUMMARY['Q3_sessions'] = st.assign(first=st['first'].astype(str), last=st['last'].astype(str)).to_dict('index')

# Class x day table: if each class was shot on different days, "day" itself is a class proxy.
cd = pd.crosstab(df.day, df['class']); cd.to_csv(OUT_DIR / 'q3_class_by_day.csv')
SUMMARY['Q3_days_with_single_class'] = int(((cd > 0).sum(1) == 1).sum())
SUMMARY['Q3_n_days'] = int(len(cd))

# %% ---------------------------------------------------------------- Q2 duplicates
import imagehash
df['phash'] = [imagehash.phash(ImageOps.exif_transpose(Image.open(p)).convert('RGB')) for p in df.path]

# ImageNet embeddings (frozen, not fine-tuned on this data, so no leakage from our own model)
import torch, timm
from torchvision import transforms
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0).eval().to(dev)
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
E = []
with torch.no_grad():
    for i in range(0, len(df), 64):
        x = torch.stack([TF(ImageOps.exif_transpose(Image.open(p)).convert('RGB')) for p in df.path[i:i + 64]])
        E.append(torch.nn.functional.normalize(enc(x.to(dev)), dim=1).cpu())
E = torch.cat(E).numpy(); np.save(OUT_DIR / 'embeddings_imagenet_b0.npy', E)

H = np.array([np.asarray(h.hash, dtype=np.uint8).ravel() for h in df.phash])
pairs = []
for i in range(len(df) - 1):
    ham = (H[i] != H[i + 1:]).sum(1)
    cos = E[i + 1:] @ E[i]
    for j in np.where((ham <= PHASH_THR) | (cos >= EMB_THR))[0]:
        pairs.append((i, i + 1 + j, int(ham[j]), float(cos[j])))
pairs = pd.DataFrame(pairs, columns=['i', 'j', 'hamming', 'cosine'])
pairs['class_i'] = df['class'].values[pairs.i]; pairs['class_j'] = df['class'].values[pairs.j]
pairs['file_i'] = df.file.values[pairs.i]; pairs['file_j'] = df.file.values[pairs.j]
pairs.to_csv(OUT_DIR / 'q2_near_duplicate_pairs.csv', index=False)

# Union-find clusters of near-duplicates -> 'dup_cluster' (a minimal grouping for splitting)
parent = list(range(len(df)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for i, j in zip(pairs.i, pairs.j):
    parent[find(i)] = find(j)
df['dup_cluster'] = [find(i) for i in range(len(df))]
csz = df.dup_cluster.value_counts()
SUMMARY['Q2'] = {
    'exact_duplicate_files': int(df.md5.duplicated(keep=False).sum()),
    'exact_duplicates_cross_class': int(df.groupby('md5')['class'].nunique().gt(1).sum()),
    'near_dup_pairs': int(len(pairs)),
    'near_dup_pairs_cross_class': int((pairs.class_i != pairs.class_j).sum()),
    'images_in_a_dup_cluster': int((df.dup_cluster.map(csz) > 1).sum()),
    'largest_cluster': int(csz.max()), 'n_clusters_size_gt1': int((csz > 1).sum()),
}
print(SUMMARY['Q2'])

# Contact sheet of the 12 largest clusters to eyeball (are they the same fruit? augmented copies?)
top = csz[csz > 1].index[:12]
if len(top):
    fig, axes = plt.subplots(len(top), 6, figsize=(12, 2.1 * len(top)))
    axes = np.atleast_2d(axes)
    for r, cl in enumerate(top):
        idx = df.index[df.dup_cluster == cl][:6]
        for c in range(6):
            ax = axes[r, c]; ax.axis('off')
            if c < len(idx):
                ax.imshow(ImageOps.exif_transpose(Image.open(df.path[idx[c]])).convert('RGB').resize((160, 160)))
                ax.set_title(df['class'][idx[c]][:10], fontsize=9)
    savefig(fig, 'q2_duplicate_clusters_contact_sheet')

# %% ---------------------------------------------------------------- Q4 shortcut checks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score


def hsv_hist(im, bins=8):
    a = np.asarray(im.convert('HSV')).reshape(-1, 3)
    h = [np.histogram(a[:, k], bins=bins, range=(0, 255), density=True)[0] for k in range(3)]
    return np.concatenate(h)


def shortcut_feats(p):
    im = ImageOps.exif_transpose(Image.open(p)).convert('RGB').resize((128, 128))
    a = np.asarray(im)
    border = np.concatenate([a[:16].reshape(-1, 3), a[-16:].reshape(-1, 3),
                             a[:, :16].reshape(-1, 3), a[:, -16:].reshape(-1, 3)])  # mostly background
    centre = a[40:88, 40:88]                                                           # mostly fruit
    g = np.asarray(im.convert('L'), dtype=np.float32)
    return {'border': hsv_hist(Image.fromarray(border.reshape(-1, 16, 3).astype(np.uint8))),
            'centre': hsv_hist(Image.fromarray(centre)),
            'global_stats': np.array([g.mean(), g.std(), np.abs(np.diff(g, axis=1)).mean()]),  # brightness/contrast/sharpness
            'thumb8': np.asarray(im.resize((8, 8))).ravel() / 255.0}


F = [shortcut_feats(p) for p in df.path]
y = df.label.values
feature_sets = {
    'border_hsv (background only)': np.stack([f['border'] for f in F]),
    'centre_hsv (fruit only)': np.stack([f['centre'] for f in F]),
    'brightness/contrast/sharpness': np.stack([f['global_stats'] for f in F]),
    'image_meta (w,h,bytes)': df[['width', 'height', 'bytes']].values.astype(float),
    '8x8 thumbnail': np.stack([f['thumb8'] for f in F]),
    'ImageNet embedding (frozen)': E,
}


def cv_bacc(X, groups=None):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=SEED) if groups is not None
          else StratifiedKFold(5, shuffle=True, random_state=SEED))
    pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    return balanced_accuracy_score(y, pred)


# Grouping used for the leakage comparison: session if timestamps exist, else duplicate cluster
group_key = 'session' if df.time.notna().mean() > 0.8 else 'dup_cluster'
groups = df[group_key].astype(str).values
res = []
for name, X in feature_sets.items():
    r = {'features': name, 'random_cv_bacc': cv_bacc(X)}
    r[f'grouped_cv_bacc ({group_key})'] = cv_bacc(X, groups)
    res.append(r); print(r)
res = pd.DataFrame(res); res.to_csv(OUT_DIR / 'q4_q5_shortcut_and_leakage.csv', index=False)
SUMMARY['Q4_Q5'] = res.to_dict('records'); SUMMARY['Q4_Q5_group_key'] = group_key
SUMMARY['chance_bacc'] = 0.2

# %% ---------------------------------------------------------------- figures (one file per panel)
# (a) class counts
fig, ax = plt.subplots(figsize=(6.5, 4))
vc = df['class'].value_counts().reindex(CLASSES)
ax.bar(range(5), vc.values, color=CB[:5]); ax.set_xticks(range(5))
ax.set_xticklabels(['Alt.', 'Anth.', 'B. blight', 'Cerc.', 'Healthy']); ax.set_ylabel('Images')
savefig(fig, 'fig_a_class_counts')

# (b) images per capture day, stacked by class
if df.time.notna().any():
    fig, ax = plt.subplots(figsize=(9, 4))
    cdp = cd.drop(index='unknown', errors='ignore')
    bottom = np.zeros(len(cdp))
    for k, c in enumerate(CLASSES):
        if c in cdp:
            ax.bar(range(len(cdp)), cdp[c].values, bottom=bottom, color=CB[k], label=c.replace('_', ' '))
            bottom += cdp[c].values
    ax.set_xlabel('Capture day (index)'); ax.set_ylabel('Images'); ax.legend(frameon=False, fontsize=11)
    savefig(fig, 'fig_b_images_per_day')

# (c) random vs grouped CV for each shortcut feature set
fig, ax = plt.subplots(figsize=(7, 4.5))
xx = np.arange(len(res)); wdt = 0.38
ax.barh(xx - wdt / 2, res.iloc[:, 1], wdt, color=CB[0], label='Random CV')
ax.barh(xx + wdt / 2, res.iloc[:, 2], wdt, color=CB[1], label='Grouped CV')
ax.axvline(0.2, color='k', ls='--', lw=1); ax.set_yticks(xx); ax.set_yticklabels(res.features, fontsize=12)
ax.set_xlabel('Balanced accuracy'); ax.set_xlim(0, 1); ax.legend(frameon=False, fontsize=11)
savefig(fig, 'fig_c_shortcut_random_vs_grouped')

# (d) t-SNE of frozen embeddings coloured by class, (e) same coloured by capture day
from sklearn.manifold import TSNE
T = TSNE(2, random_state=SEED, init='pca', perplexity=30).fit_transform(E)
fig, ax = plt.subplots(figsize=(6, 6))
for k, c in enumerate(CLASSES):
    m = df.label.values == k; ax.scatter(T[m, 0], T[m, 1], s=4, color=CB[k], label=c.replace('_', ' '))
ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, markerscale=4, fontsize=11)
savefig(fig, 'fig_d_tsne_by_class')
if df.time.notna().any():
    fig, ax = plt.subplots(figsize=(6, 6))
    days = df.day.astype('category').cat.codes.values
    ax.scatter(T[:, 0], T[:, 1], s=4, c=days, cmap='viridis'); ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, 'fig_e_tsne_by_day')

# (f) random sample grid per class (check backgrounds, stamps, framing)
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(5, 6, figsize=(12, 10.5))
for r, c in enumerate(CLASSES):
    pick = rng.choice(df.index[df['class'] == c], 6, replace=False)
    for k, i in enumerate(pick):
        axes[r, k].imshow(ImageOps.exif_transpose(Image.open(df.path[i])).convert('RGB').resize((200, 200)))
        axes[r, k].axis('off')
    axes[r, 0].set_title(c.replace('_', ' '), loc='left', fontsize=12)
savefig(fig, 'fig_f_sample_grid')

# %% ---------------------------------------------------------------- save
df.drop(columns=['phash']).assign(time=df.time.astype(str)).to_csv(OUT_DIR / 'audit_per_image.csv', index=False)
df[['path', group_key]].rename(columns={group_key: 'group'}).to_csv(OUT_DIR / 'groups.csv', index=False)
json.dump(SUMMARY, open(OUT_DIR / 'summary.json', 'w'), indent=1, default=str)
print(json.dumps(SUMMARY, indent=1, default=str))
print('audit written to', OUT_DIR)
