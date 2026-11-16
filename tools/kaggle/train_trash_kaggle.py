# %% [markdown]
# # Robodog trash detector: fine-tune YOLO on open litter data + our robot recordings (Kaggle)
#
# What this notebook does (one "Run all", ~1-1.5 h on a Kaggle T4):
# 1. **Our hand-labelled robot frames** (Roboflow export, YOLO format): found automatically by their
#    file names (they start with the recording's date_time). The split follows the VIDEO: frames of
#    the newest recording are validation only, whatever split Roboflow made.
#    (Optional, off by default: `USE_PSEUDO_LABELS = True` also auto-labels the videos with the
#    zero-shot detector Grounding DINO. Hand labels are much better.)
# 2. **Converts open datasets** to YOLO format, everything as ONE class (`CLASS_NAME`, "litter"):
#    TACO (COCO json) and any other YOLO-format dataset you attach.
# 3. **Trains YOLO11s at 1280 px** (small trash at 2-4 m needs the resolution).
#    Validation = our most recent recording (never seen in training) + 10 % of the open data.
# 4. **Renders our videos with the trained model** (boxes + track IDs, like robodog-trash), as
#    mp4 files in the Output tab, and shows a short clip here.
#
# ## Setup on Kaggle (once)
# 1. kaggle.com -> sign in -> verify your phone (needed for GPU + internet).
# 2. **Upload our recordings** as a dataset: Datasets -> New Dataset -> drag the 3 recording folders
#    (each has a `video.mkv`) -> name it e.g. `robodog-recordings` -> Create (private).
# 3. New Notebook -> File -> Import Notebook -> this file (or paste the cells).
# 4. Right panel: **Add Input** -> your `robodog-recordings` (videos, for the rendered results) and your
#    **Roboflow export** (upload the zip as a dataset: Format YOLOv11/YOLOv8, NO resize preprocessing);
#    Add Input -> search **"TACO"**
#    (e.g. "TACO Trash Dataset", the one with `annotations.json` + `batch_*` folders).
#    Optional: add more YOLO-format litter datasets (folders with `images/` and `labels/`).
# 5. Right panel: **Accelerator: GPU T4** (x2 is fine, we use one), **Internet: on**.
# 6. **Save Version -> Save & Run All (Commit)**: runs in the background, you can close the tab.
#    Results: the version's **Output** tab (`trash_yolo11s_1280.pt`, `videos/*.mp4`, previews).
#
# ## Pause and continue (training saves `runs/trash/weights/last.pt` after EVERY epoch)
# - **Same session** (interactive): press Stop (the square) on the training cell, change settings if
#   you like (`EPOCHS`, `WORKERS`, `BATCH`, `PATIENCE`), run the settings cell and the training cell
#   again: it continues from the last finished epoch (`RESUME = True`).
# - **New session**: download `runs/trash/weights/last.pt` from the Output panel first (the files
#   are gone when the session ends), upload it as a dataset, add it as input, set `RESUME_FROM` to
#   its path (e.g. "/kaggle/input/robodog-last/last.pt") and Run all.
#
# ## Use the model on the laptop
# Download `trash_yolo11s_1280.pt` -> put it in the repo, e.g. `models/` -> in `config.local.yaml`:
# `trash.model: models/trash_yolo11s_1280.pt`, `trash.imgsz: 1280`, `trash.classes: ""` (or "litter").
# Then: `uv run robodog-trash recordings/<run>/video.mkv`.

# %%
# ---- settings (the defaults are fine for a first run) ----
CLASS_NAME = "litter"      # the ONE class the model learns (every label from every dataset becomes this)
USE_PSEUDO_LABELS = False  # True: ALSO auto-label the videos with Grounding DINO (the settings below)
SAMPLE_EVERY = 10          # take every 10th video frame (30 fps -> 3 frames/s)
LABEL_THRESHOLD = 0.30     # Grounding DINO confidence for a pseudo-label (higher = fewer, surer boxes)
PROMPTS = ["piece of paper", "plastic bottle", "can", "cup", "wrapper", "cardboard box",
           "plastic bag", "crumpled paper", "plastic cup", "bottle cap", "cigarette pack", "trash"]
# Our camera looks slightly UP: the floor starts at ~73 % of the image height (things on tables,
# under tables and on people end above that). Check the previews: if real floor trash is cut, lower these.
FLOOR_FROM = 0.55          # only search below this fraction of the frame height
MIN_BOTTOM = 0.74          # a box must end below this fraction (= it lies on the floor, not on a table)
MAX_AREA = 0.03            # drop boxes bigger than 3 % of the frame (from this camera trash is small)
EMPTY_KEEP = 0.10          # keep this share of frames with no trash found (as "background")
OPEN_MAX_IMAGES = 1500     # cap on open-dataset images in total (fewer = faster epochs)
OPEN_MAX_PER_SOURCE = 800  # ...and per dataset, so one big dataset doesn't crowd out the others
OWN_REPEAT = 3             # our hand-labelled training frames are used this many times (more weight
                           # for our camera's view next to thousands of public images)
# Classes in public datasets that are NOT trash (matched as words in the dataset's class names):
SKIP_CLASS_WORDS = ["person", "people", "human", "hand", "face", "bin", "container", "dumpster",
                    "robot", "animal", "dog", "cat", "car", "vehicle", "tree", "plant"]
OPEN_MAX_SIDE = 1280       # open-dataset images are resized to at most this
MODEL = "yolo11s.pt"       # yolo11n.pt = faster on the laptop CPU later, yolo11m.pt = more accurate
IMGSZ = 1280
EPOCHS = 40                # it usually stops improving well before; PATIENCE ends it early
PATIENCE = 10              # stop when validation hasn't improved for this many epochs
BATCH = 8                  # T4 16 GB: 8 at 1280 for yolo11s; lower if "CUDA out of memory"
WORKERS = 3                # image-loading processes (Kaggle has 4 CPUs); each holds 1280 px mosaics in RAM.
                           # Out of RAM -> 2 or 0. GPU meter low + CPU meter full -> loading is the bottleneck
TRAIN_RETRIES = 2          # if training dies (e.g. out of RAM), resume from the last checkpoint this often
RESUME = True              # an unfinished training in runs/trash: continue it (False: start over)
RESUME_FROM = ""           # a last.pt from an earlier session (uploaded as input), e.g. "/kaggle/input/x/last.pt"
VAL_VIDEOS = 1             # the newest N recordings are used for validation only
RENDER_CONF = 0.25         # confidence for the rendered videos
GD_NAME = "IDEA-Research/grounding-dino-base"

import os
if os.environ.get("TRASH_SMOKE"):   # quick local test of the whole notebook (not for Kaggle)
    MODEL, IMGSZ, BATCH, OPEN_MAX_IMAGES = "yolo11n.pt", 320, 4, 20
    EPOCHS = int(os.environ.get("TRASH_SMOKE_EPOCHS", 1))
    GD_NAME = "IDEA-Research/grounding-dino-tiny"

# %%
import glob, json, os, random, re, shutil, subprocess, sys
from pathlib import Path

if not os.environ.get("TRASH_SMOKE"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "lap"], check=True)

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

random.seed(0)
IN = Path(os.environ.get("KAGGLE_IN", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working"))
DS = Path(os.environ.get("KAGGLE_TEMP", "/kaggle/temp")) / "dataset"   # big intermediate files: not in the output
OUT = WORK
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    (DS / sub).mkdir(parents=True, exist_ok=True)
(OUT / "previews").mkdir(exist_ok=True)
(OUT / "videos").mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ROBOT_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")   # our recordings' date_time in a file name


def yolo_line(box, W, H):
    x0, y0, x1, y1 = box
    return f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"


def is_skipped_class(name):
    words = re.split(r"[^a-z]+", str(name).lower())
    return any(w in words for w in SKIP_CLASS_WORDS)


def dataset_classes(label_dir):
    """Class names from the data.yaml next to a YOLO dataset (Roboflow: <root>/data.yaml), or {}."""
    import yaml
    for d in (label_dir.parent, label_dir.parent.parent, label_dir.parent.parent.parent):
        for f in ("data.yaml", "dataset.yaml"):
            if (d / f).exists():
                names = (yaml.safe_load((d / f).read_text()) or {}).get("names", {})
                return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}, d
    return {}, label_dir.parent


def image_for_label(txt, label_dir):
    """The image belonging to a YOLO label file (labels/x/y.txt -> images/x/y.<ext>). Names may contain
    extra dots (Roboflow: '..._jpg.rf.abc123.txt'), so only the final '.txt' is replaced."""
    rel_dir = txt.relative_to(label_dir).parent
    base = txt.name[: -len(".txt")]
    return next((p for e in IMG_EXT for p in [label_dir.parent / "images" / rel_dir / (base + e)] if p.exists()), None)


def read_yolo_boxes(txt, W, H, skip_ids=()):
    """YOLO label file -> pixel boxes (any class = trash, except skip_ids). Polygons become their box."""
    boxes = []
    for line in txt.read_text().split("\n"):
        p = line.split()
        if p and p[0].isdigit() and int(p[0]) in skip_ids:
            continue
        if len(p) == 5:
            cx, cy, bw, bh = map(float, p[1:5])
            boxes.append(((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        elif len(p) > 5 and len(p) % 2 == 1:
            xs, ys = [float(v) * W for v in p[1::2]], [float(v) * H for v in p[2::2]]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: switch the accelerator on!)")

# %% [markdown]
# ## 1. Find the inputs

# %%
videos = sorted(p for ext in ("mkv", "mp4", "avi", "mov") for p in IN.rglob(f"*.{ext}"))
# newest last: recording folders are named YYYY-MM-DD_HH-MM-SS, else sort by path
videos.sort(key=lambda p: (p.parent.name, str(p)))
print(f"{len(videos)} videos:", *[f"  {v}" for v in videos], sep="\n")
assert videos, "no videos found: add your recordings dataset as an input"
val_videos = set(videos[-VAL_VIDEOS:]) if len(videos) > VAL_VIDEOS else set()
print("validation videos:", [str(v) for v in val_videos])

taco_jsons = [p for p in IN.rglob("*.json")
              if p.stat().st_size > 100_000 and '"categories"' in p.open(errors="ignore").read(5_000_000)]
print("COCO-style annotation files (TACO etc.):", [str(p) for p in taco_jsons])
yolo_label_dirs = [p for p in IN.rglob("labels") if p.is_dir() and any(p.rglob("*.txt"))]
print("YOLO-format label folders:", [str(p) for p in yolo_label_dirs])

# %% [markdown]
# ## 2. Our hand-labelled robot frames (Roboflow export)

# %%
val_names = {v.parent.name for v in val_videos}
own_items = []
for ld in yolo_label_dirs:
    for txt in ld.rglob("*.txt"):
        img = image_for_label(txt, ld)
        m = ROBOT_NAME.search(txt.name)
        if img is None or not m:
            continue
        # Split by VIDEO (honest validation); frames of an unknown recording keep Roboflow's split.
        if m.group(0) in val_names:
            split = "val"
        elif val_names:
            split = "train"
        else:
            split = "val" if ld.parent.name in ("valid", "val", "test") else "train"
        own_items.append((img, txt, split, m.group(0)))

own_stats, sizes, own_preview = {"train": [0, 0], "val": [0, 0]}, {}, []
for img, txt, split, rec in own_items:
    im = cv2.imread(str(img))
    if im is None:
        print("unreadable:", img); continue
    H, W = im.shape[:2]
    sizes[(W, H)] = sizes.get((W, H), 0) + 1
    boxes = read_yolo_boxes(txt, W, H)
    label = "\n".join(yolo_line(b, W, H) for b in boxes)
    f = min(1.0, IMGSZ / max(W, H))           # training resizes to IMGSZ anyway: store at that size
    small = cv2.resize(im, (round(W * f), round(H * f)), interpolation=cv2.INTER_AREA) if f < 1 else im
    first = None
    for k in range(OWN_REPEAT if split == "train" else 1):
        name = f"own_{img.stem}" + (f"_r{k}" if k else "")
        dst = DS / f"images/{split}/{name}.jpg"
        if first is None:
            cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 95]); first = dst
        else:
            shutil.copy(first, dst)
        (DS / f"labels/{split}/{name}.txt").write_text(label)
    del im, small
    own_stats[split][0] += 1; own_stats[split][1] += len(boxes)
    if boxes:
        own_preview.append((img, boxes, f"own_{img.stem}"))   # path only: full frames would eat ~2 GB RAM
print(f"hand-labelled robot frames / boxes (training frames used {OWN_REPEAT}x):", own_stats, "  recordings:", sorted({r for *_, r in own_items}))
print("image sizes:", sizes)
if not own_items:
    print("!! no hand-labelled robot frames found: add the Roboflow export as an input "
          "(file names must still contain the recording's date_time, e.g. 2026-11-16_01-58-30_0001...)")
if any(w < 1500 for w, h in sizes):
    print("!! some frames are smaller than the 1920x1080 originals: was 'Resize' left on in Roboflow? "
          "Small distant trash suffers; re-export without resize if possible.")

# %%
# Check that the labels were read correctly (red boxes = your Roboflow labels).
import matplotlib.pyplot as plt

picks = random.sample(own_preview, min(12, len(own_preview)))
if picks:
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    for ax, (path, boxes, name) in zip(axes.flat, picks):
        img = cv2.imread(str(path))
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 3)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_title(name, fontsize=8); ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "previews" / "own_labels_grid.jpg", dpi=60); plt.show()
    plt.close("all")

# %% [markdown]
# ## 2b. Optional: auto-label the videos with Grounding DINO (USE_PSEUDO_LABELS)

# %%
if USE_PSEUDO_LABELS:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from torchvision.ops import nms

    gd_proc = AutoProcessor.from_pretrained(GD_NAME)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_NAME).to(DEVICE).eval()
    PROMPT = " . ".join(p.lower() for p in PROMPTS) + " ."
    _ids = gd_proc.tokenizer(PROMPT)["input_ids"]
    _skip = {gd_proc.tokenizer.cls_token_id, gd_proc.tokenizer.sep_token_id,
             gd_proc.tokenizer.convert_tokens_to_ids(".")}
    TOKEN_MASK = torch.tensor([i not in _skip for i in _ids], device=DEVICE)


    @torch.inference_mode()
    def gd_detect(rgb):
        """Boxes (x0,y0,x1,y1 px) and scores in this RGB crop: any prompt word counts as 'trash'."""
        h, w = rgb.shape[:2]
        inputs = gd_proc(images=rgb, text=PROMPT, return_tensors="pt").to(DEVICE)
        out = gd_model(**inputs)
        probs = out.logits[0].sigmoid()[:, : len(_ids)]
        scores = (probs * TOKEN_MASK).max(dim=-1).values
        keep = scores > LABEL_THRESHOLD
        cx, cy, bw, bh = out.pred_boxes[0][keep].T
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dim=1)
        return boxes.cpu().numpy(), scores[keep].cpu().numpy()


    def label_frame(bgr):
        """Floor band of a 1080p frame in 2 overlapping tiles (near native resolution), merged."""
        H, W = bgr.shape[:2]
        top = int(H * FLOOR_FROM)
        band = cv2.cvtColor(bgr[top:], cv2.COLOR_BGR2RGB)
        tiles = [(0, int(W * 0.55)), (int(W * 0.45), W)]          # 10 % overlap in the middle
        all_b, all_s = [], []
        for x0, x1 in tiles:
            b, s = gd_detect(np.ascontiguousarray(band[:, x0:x1]))
            if len(b):
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += top
                all_b.append(b); all_s.append(s)
        if not all_b:
            return []
        b, s = np.concatenate(all_b), np.concatenate(all_s)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
        keep = nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), 0.5).numpy()
        res = []
        for x0, y0, x1, y1 in b[keep].tolist():
            bw, bh = x1 - x0, y1 - y0
            if y1 < MIN_BOTTOM * H or bw * bh > MAX_AREA * W * H or bw < 6 or bh < 6:
                continue
            res.append((x0, y0, x1, y1))
        return res


    stats = {"train": [0, 0], "val": [0, 0]}      # [frames, boxes]
    preview = []
    for vid in videos:
        split = "val" if vid in val_videos else "train"
        cap = cv2.VideoCapture(str(vid))
        i, kept = -1, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % SAMPLE_EVERY:
                continue
            boxes = label_frame(frame)
            if not boxes and random.random() > EMPTY_KEEP:
                continue
            H, W = frame.shape[:2]
            name = f"robot_{vid.parent.name}_{i:05d}"
            cv2.imwrite(str(DS / f"images/{split}/{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (DS / f"labels/{split}/{name}.txt").write_text("\n".join(yolo_line(b, W, H) for b in boxes))
            stats[split][0] += 1; stats[split][1] += len(boxes); kept += 1
            if boxes:
                preview.append((DS / f"images/{split}/{name}.jpg", boxes, name))   # path only (RAM)
        cap.release()
        print(f"{vid.parent.name}: {i + 1} frames read, {kept} kept ({split})")
    print("robot frames / boxes:", stats)

# %%
if USE_PSEUDO_LABELS:
    # Preview of the pseudo-labels: CHECK THESE. Too many wrong boxes -> raise LABEL_THRESHOLD;
    # missed trash -> lower it (or label by hand later, see the end of the notebook).
    import matplotlib.pyplot as plt

    picks = random.sample(preview, min(12, len(preview)))
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    for ax, (path, boxes, name) in zip(axes.flat, picks):
        img = cv2.imread(str(path))
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.imwrite(str(OUT / "previews" / f"{name}.jpg"), img)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_title(name, fontsize=8); ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "previews" / "pseudo_labels_grid.jpg", dpi=60); plt.show()
    plt.close("all")

    del gd_model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Open datasets -> YOLO, one class (CLASS_NAME)

# %%
def save_open(img, boxes_xyxy, name):
    """Resize to OPEN_MAX_SIDE, 90 % train / 10 % val, write image + YOLO label."""
    W, H = img.size
    k = min(1.0, OPEN_MAX_SIDE / max(W, H))
    if k < 1:
        img = img.resize((round(W * k), round(H * k)), Image.BILINEAR)
    split = "val" if random.random() < 0.1 else "train"
    img.convert("RGB").save(DS / f"images/{split}/{name}.jpg", quality=90)
    lines = [yolo_line(b, W, H) for b in boxes_xyxy if b[2] - b[0] > 1 and b[3] - b[1] > 1]
    (DS / f"labels/{split}/{name}.txt").write_text("\n".join(lines))
    return split


open_items = []   # (kind, path, size, annotation, name, source)

# COCO json (TACO): every annotation = trash (except SKIP_CLASS_WORDS categories)
for jf in taco_jsons:
    coco = json.loads(jf.read_text())
    root = jf.parent
    skip_cats = {c["id"] for c in coco.get("categories", []) if is_skipped_class(c.get("name", ""))}
    if skip_cats:
        print(f"{jf}: skipping classes", [c["name"] for c in coco["categories"] if c["id"] in skip_cats])
    by_img = {}
    for a in coco.get("annotations", []):
        if a.get("category_id") in skip_cats:
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    for im in coco.get("images", []):
        path = root / im["file_name"]
        if not path.exists():
            hits = list(root.rglob(Path(im["file_name"]).name))
            if not hits:
                continue
            path = hits[0]
        open_items.append(("coco", path, (im["width"], im["height"]), by_img.get(im["id"], []),
                           f"coco_{jf.parent.name}_{im['id']}", str(jf)))

# YOLO-format folders (other open datasets): every class = trash, except SKIP_CLASS_WORDS.
# Our own frames were added in step 2.
skip_by_dir = {}
for ld in yolo_label_dirs:
    classes, root = dataset_classes(ld)
    skip = {i for i, n in classes.items() if is_skipped_class(n)}
    skip_by_dir[ld] = skip
    if classes:
        print(f"{root.name}/{ld.parent.name}: classes {list(classes.values())[:12]}"
              + (f", skipping {[classes[i] for i in sorted(skip)]}" if skip else ""))
    for txt in ld.rglob("*.txt"):
        if ROBOT_NAME.search(txt.name):
            continue
        img = image_for_label(txt, ld)
        if img is not None:
            rel = txt.relative_to(ld).with_name(txt.name[: -len(".txt")])
            open_items.append(("yolo", img, None, (txt, ld), f"yolo_{root.name}_{ld.parent.name}_{'_'.join(rel.parts)}",
                               str(root)))

# Per-dataset cap, then the total cap.
by_source = {}
for it in open_items:
    by_source.setdefault(it[5], []).append(it)
open_items = []
for src, items in by_source.items():
    random.shuffle(items)
    print(f"open dataset {src}: {len(items)} images, using {min(len(items), OPEN_MAX_PER_SOURCE)}")
    open_items += items[:OPEN_MAX_PER_SOURCE]
random.shuffle(open_items)
open_items = open_items[:OPEN_MAX_IMAGES]
n_open = {"train": 0, "val": 0}
for kind, path, size, ann, name, src in open_items:
    try:
        img = Image.open(path)
        if kind == "coco":
            if img.size != tuple(size):
                img = ImageOps.exif_transpose(img)   # TACO: boxes are for the upright photo
            if img.size != tuple(size):
                continue
            boxes = ann
        else:
            img = ImageOps.exif_transpose(img)
            txt, ld = ann
            boxes = read_yolo_boxes(txt, *img.size, skip_ids=skip_by_dir[ld])
        n_open[save_open(img, boxes, name)] += 1
    except Exception as e:
        print("skip", path, e)
print("open-dataset images:", n_open)

# Training needs a validation set: if none came from anywhere, move 10 % of the training images there.
if not any((DS / "images/val").iterdir()):
    train_imgs = sorted((DS / "images/train").iterdir())
    for im in random.sample(train_imgs, max(1, len(train_imgs) // 10)):
        shutil.move(im, DS / "images/val" / im.name)
        lbl = DS / "labels/train" / (im.stem + ".txt")
        if lbl.exists():
            shutil.move(lbl, DS / "labels/val" / lbl.name)
    print("!! no validation images: moved 10 % of the training images to validation")
print("final dataset:", {sp: len(list((DS / f"images/{sp}").iterdir())) for sp in ("train", "val")})

(DS / "data.yaml").write_text(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAME}\n")
print((DS / "data.yaml").read_text())

# %% [markdown]
# ## 4. Train

# %%
# Training runs in a SEPARATE Python process: the notebook's own memory (previews, dataset lists)
# isn't added to it, and if it dies (out of RAM) the notebook survives and resumes from the last
# checkpoint (last.pt, saved after every epoch).
# Output: the running progress bar stays on ONE line that updates; only finished lines are kept
# (one line per epoch + the validation scores), with the time per epoch and the time left.
# To PAUSE: Stop this cell. To continue: run it again (see "Pause and continue" at the top).
import gc, time
import matplotlib.pyplot as plt
from ultralytics import YOLO

gc.collect()
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM before training: {vm.used / 1e9:.1f} GB used of {vm.total / 1e9:.1f} GB")
except ImportError:
    pass

run_dir = WORK / "runs/trash"
last, best = run_dir / "weights/last.pt", run_dir / "weights/best.pt"
train_args = dict(data=str(DS / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
                  device=0 if DEVICE == "cuda" else "cpu", patience=PATIENCE, degrees=0, flipud=0.0, fliplr=0.5,
                  scale=0.5, mosaic=1.0, close_mosaic=10, hsv_v=0.5, project=str(WORK / "runs"), name="trash",
                  exist_ok=True, plots=True, workers=WORKERS, cache=False, save_period=5)
# settings that may change when continuing (the others stay as the run started)
resume_args = dict(data=str(DS / "data.yaml"), batch=BATCH, workers=WORKERS, patience=PATIENCE,
                   device=train_args["device"], save_period=5)

if RESUME_FROM and not last.exists():
    last.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(RESUME_FROM, last)
    print("continuing from", RESUME_FROM)


def checkpoint_state():
    """'none' / 'finished' / 'unfinished' (and then sets that checkpoint's total epochs to EPOCHS)."""
    if not last.exists():
        return "none"
    ck = torch.load(last, map_location="cpu", weights_only=False)
    done = ck.get("epoch", -1) + 1            # epochs finished; -1 = the run completed
    if done <= 0 or done >= EPOCHS:
        return "finished"
    if ck["train_args"].get("epochs") != EPOCHS:
        print(f"total epochs of the saved run: {ck['train_args'].get('epochs')} -> {EPOCHS}")
        ck["train_args"]["epochs"] = EPOCHS   # the only way to change it on resume
        torch.save(ck, last)
    print(f"continuing the unfinished training: {done} of {EPOCHS} epochs done")
    return "unfinished"


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Ultralytics' own per-epoch lines: replaced by ONE summary row per epoch (from results.csv)
EPOCH_NOISE = re.compile(r"^\s*(\d+/\d+\s+\S+G\s|Epoch\s+GPU_mem|Class\s+Images|all\s+\d+\s+\d+)")
COLS = [("train/box_loss", "box"), ("train/cls_loss", "cls"), ("train/dfl_loss", "dfl"),
        ("val/box_loss", "box"), ("val/cls_loss", "cls"), ("val/dfl_loss", "dfl"),
        ("metrics/precision(B)", "P"), ("metrics/recall(B)", "R"),
        ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]
HEADER = (f"{'epoch':>9} | {'train loss: box  cls   dfl':^26} | {'val loss: box  cls   dfl':^24} | "
          f"{'P':>5} {'R':>5} {'mAP50':>6} {'mAP50-95':>8} | time")


def read_results():
    f = run_dir / "results.csv"
    if not f.exists():
        return []
    import csv
    with open(f) as fh:
        return [{k.strip(): v.strip() for k, v in row.items() if k} for row in csv.DictReader(fh)]


def summary_row(rows, i):
    r = rows[i]
    v = [float(r.get(k, "nan") or "nan") for k, _ in COLS]
    t = float(r.get("time", 0) or 0) - (float(rows[i - 1].get("time", 0) or 0) if i else 0.0)
    left = (EPOCHS - int(float(r["epoch"]))) * t
    return ((f"{int(float(r['epoch'])):>4}/{EPOCHS:<4} | {v[0]:8.3f} {v[1]:5.2f} {v[2]:5.3f}        | "
            f"{v[3]:7.3f} {v[4]:5.2f} {v[5]:5.3f}      | {v[6]:5.3f} {v[7]:5.3f} {v[8]:6.3f} {v[9]:8.3f} | ")
            + (f"{t / 60:.1f} min, ~{left / 3600:.1f} h left at most" if t > 0 else ""))


def plot_curves(rows):
    """Train vs validation loss + mAP per epoch: falling train loss with RISING val loss = overfitting."""
    ep = [int(float(r["epoch"])) for r in rows]
    col = lambda k: [float(r.get(k, "nan") or "nan") for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for a, name in zip(ax, ("box_loss", "cls_loss", "dfl_loss")):
        a.plot(ep, col(f"train/{name}"), "o-", ms=3, label="train")
        a.plot(ep, col(f"val/{name}"), "o-", ms=3, label="validation (robot video)")
        a.set_title(name); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[3].plot(ep, col("metrics/mAP50(B)"), "o-", ms=3, label="mAP50")
    ax[3].plot(ep, col("metrics/mAP50-95(B)"), "o-", ms=3, label="mAP50-95")
    ax[3].plot(ep, col("metrics/recall(B)"), "--", label="recall")
    ax[3].set_title("validation scores (higher = better)"); ax[3].set_xlabel("epoch"); ax[3].grid(alpha=0.3)
    ax[3].legend(fontsize=8)
    plt.tight_layout()
    (OUT / "previews").mkdir(exist_ok=True)
    path = OUT / "previews" / "training_curves.png"
    fig.savefig(path, dpi=70); plt.close(fig)
    return path


def run_training(code):
    """Run one training process. Returns its exit code. Stop the cell = stop training (keeps last.pt)."""
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1"))
    try:
        from IPython import get_ipython
        from IPython.display import Image, Pretty, display
        if get_ipython() is None:
            raise ImportError
        live = display(Pretty(""), display_id=True)
        chart = display(Pretty("(loss chart appears after the first epoch)"), display_id=True)
    except ImportError:
        live = chart = None
    state = {"shown": "", "t": 0.0, "rows": 0, "header": False}

    def show(text, force=False):
        if text == state["shown"] or (not force and time.time() - state["t"] < 0.5):
            return
        state["shown"], state["t"] = text, time.time()
        if live is not None:
            live.update(Pretty(text))

    def new_epochs():
        rows = read_results()
        if len(rows) <= state["rows"]:
            return
        if not state["header"]:
            print(HEADER, flush=True); state["header"] = True
        for i in range(state["rows"], len(rows)):
            print(summary_row(rows, i), flush=True)
        state["rows"] = len(rows)
        png = plot_curves(rows)
        if chart is not None:
            chart.update(Image(filename=str(png)))

    buf = ""
    try:
        while chunk := proc.stdout.read1(65536):
            buf += chunk.decode("utf-8", "replace")
            # a progress bar redraws with "\r<bar>"; a finished line ends with "\n"
            *pieces, buf = re.split(r"(\r|\n)", buf)
            for text, sep in zip(pieces[0::2], pieces[1::2]):
                text = ANSI.sub("", text).rstrip()
                if sep == "\r" or EPOCH_NOISE.match(text):
                    if text:
                        show(text.strip())
                    continue
                if text:
                    print(text, flush=True)
            if buf.strip():
                show(ANSI.sub("", buf).strip())
            new_epochs()
        code = proc.wait()
        new_epochs()
        return code
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n** training stopped. last.pt keeps the last FINISHED epoch: run this cell again to continue **")
        raise
    finally:
        show("", force=True)


for attempt in range(TRAIN_RETRIES + 1):
    state = checkpoint_state() if (RESUME or attempt) else "none"
    if state == "finished" and best.exists():
        print("the training is finished: using its model" + ("" if attempt else
              " (set RESUME = False to train again from scratch)"))
        break
    if state == "unfinished":
        if attempt:
            print(f"\n!! training process died: resuming (attempt {attempt}/{TRAIN_RETRIES})\n")
        code = (f"import json; from ultralytics import YOLO; "
                f"YOLO({str(last)!r}).train(resume=True, **json.loads({json.dumps(resume_args)!r}))")
    else:
        if attempt:
            print("\n!! training died before the first checkpoint: starting again\n")
        shutil.rmtree(run_dir, ignore_errors=True)          # a clean start, no mixed-up results
        code = f"import json; from ultralytics import YOLO; YOLO({MODEL!r}).train(**json.loads({json.dumps(train_args)!r}))"
    rc = run_training(code)
    if rc == 0:
        break
    if rc in (130, -2):                     # stopped with Ctrl-C / the Stop button: don't restart it
        raise KeyboardInterrupt("training stopped: run this cell again to continue from last.pt")
else:
    if not best.exists():
        raise RuntimeError("training failed repeatedly: lower BATCH / IMGSZ or WORKERS = 0, then run this cell again")
    print("!! training did not end cleanly, but a model was saved: continuing with the best one so far")

shutil.copy(best, OUT / "trash_yolo11s_1280.pt")
print("model saved:", OUT / "trash_yolo11s_1280.pt")

# %%
# Scores on the held-out robot video (+ 10 % open data). Note: measured against the pseudo-labels.
metrics = YOLO(best).val(data=str(DS / "data.yaml"), imgsz=IMGSZ, device=0 if DEVICE == "cuda" else "cpu",
                         project=str(WORK / "runs"), name="val", exist_ok=True)
print(f"mAP50 {metrics.box.map50:.3f}   mAP50-95 {metrics.box.map:.3f}   "
      f"precision {metrics.box.mp:.3f}   recall {metrics.box.mr:.3f}")

# %% [markdown]
# ## 5. Render our videos with the trained model (boxes + track IDs)

# %%
try:
    from IPython.display import Video, display
except ImportError:                       # outside a notebook
    Video = display = None

det = YOLO(best)
for vid in videos:
    out_mp4 = OUT / "videos" / f"{vid.parent.name}_detected.mp4"
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow, oh = 1280, int(H * 1280 / W) // 2 * 2
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "26", "-pix_fmt", "yuv420p", str(out_mp4)], stdin=subprocess.PIPE)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = det.track(frame, persist=True, tracker="bytetrack.yaml", conf=RENDER_CONF, imgsz=IMGSZ,
                      verbose=False, device=0 if DEVICE == "cuda" else "cpu")[0]
        for b in r.boxes:
            x0, y0, x1, y1 = map(int, b.xyxy[0].tolist())
            tid = int(b.id) if b.id is not None else -1
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(frame, f"#{tid} {float(b.conf):.2f}", (x0, max(30, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        tag = "VAL (not trained on)" if vid in val_videos else "train video"
        cv2.putText(frame, f"{vid.parent.name}  {n / fps:5.1f} s  {tag}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        ff.stdin.write(cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA).tobytes())
        n += 1
    cap.release(); ff.stdin.close(); ff.wait()
    if det.predictor is not None and getattr(det.predictor, "trackers", None):
        det.predictor.trackers[0].reset()         # new video: fresh track IDs
    print("written", out_mp4, f"({n} frames)")

# A 20 s clip of the validation video, playable right here (full videos: Output tab -> videos/)
clip_src = OUT / "videos" / f"{sorted(val_videos or videos)[0].parent.name}_detected.mp4"
clip = OUT / "videos" / "preview_clip.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_src), "-t", "20", "-c", "copy", str(clip)], check=True)
if display:
    display(Video(str(clip), embed=True, width=960))

# %% [markdown]
# ## Next steps
# - **Check the previews** (`previews/`). If the pseudo-labels are bad, change `LABEL_THRESHOLD` and
#   run again. Best quality: correct the labels by hand. Download the dataset with the cell below,
#   fix it in CVAT / Label Studio / Roboflow (YOLO format), upload it as a Kaggle dataset, and add it
#   as an input: the notebook merges any YOLO-format folder (then lower the robot pseudo-labels'
#   weight by setting SAMPLE_EVERY high, or remove the recordings input).
# - More robot recordings with trash at 1-5 m (and trash-like non-trash) help the most.

# %%
# Optional: zip the robot part of the dataset (images + pseudo-labels) for hand correction.
# shutil.make_archive(str(OUT / "robot_pseudo_labels"), "zip", DS)
