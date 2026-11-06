"""Dataset loaders for the fheml examples.

Every loader downloads what it needs into ``data_dir`` (default ``data/``,
ignored by git), converts it once to a compact ``.npz`` cache next to it and
returns numpy arrays. Images are uint8 ``(N, C, H, W)``; the examples scale
them. Sources are mirrors that stay reachable from restricted networks
(fast.ai's S3 copies of CIFAR-10, Oxford-IIIT Pet and COCO 2017, GitHub for
CMUdict and SST-2, fast.ai's S3 copy of AG News, Google's MNIST mirror).

    mnist       handwritten digits, 28x28            classification
    cifar10     10 classes, 32x32 RGB                classification
    pets        Oxford-IIIT Pet: 37 breeds / cat-dog, pixel trimaps
                (pet, background, border) and pet head boxes
                                                     segmentation, detection
    coco        COCO 2017 instances: boxes and masks of 80 categories,
                resized to a square                  detection, segmentation
    cmudict     CMU pronouncing dictionary: spelling -> phonemes
                                                     sequence to sequence
    agnews      AG News: 4 news topics, 120k/7.6k    text classification
    sst2        Stanford Sentiment Treebank, binary  text classification

    python3 tools/datasets.py cifar10 pets coco cmudict   # fetch and cache
"""
import concurrent.futures
import hashlib
import http.client
import io
import json
import os
import struct
import sys
import tarfile
import time
import urllib.request
import zipfile
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train_mnist import load_mnist  # noqa: E402,F401  (re-exported)

FASTAI = "https://s3.amazonaws.com/fast-ai-imageclas/"
COCO = "https://s3.amazonaws.com/fast-ai-coco/"
CMUDICT = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
AGNEWS = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
SST2 = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/"

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
PET_MASK_CLASSES = ("pet", "background", "border")
AGNEWS_CLASSES = ("world", "sports", "business", "sci/tech")
SST2_CLASSES = ("negative", "positive")


def _download(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print("downloading", url, file=sys.stderr)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _fetch_tgz(url, data_dir, folder):
    """Downloads and extracts a .tgz into data_dir unless data_dir/folder exists."""
    if os.path.isdir(os.path.join(data_dir, folder)):
        return
    archive = _download(url, os.path.join(data_dir, folder + ".tgz"))
    print("extracting", archive, file=sys.stderr)
    with tarfile.open(archive) as t:
        t.extractall(data_dir, filter="data")
    os.remove(archive)


def _cached(path, build):
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    out = build()
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return out


def _image(data, size, mode="RGB"):
    """Decodes an image file and resizes it to size x size (squashed, no
    cropping): (3, size, size) uint8, or (size, size) for masks (nearest)."""
    from PIL import Image

    im = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    if mode == "L":
        return np.asarray(im.resize((size, size), Image.NEAREST), dtype=np.uint8)
    im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


# -------------------------------------------------------------------- CIFAR-10

def load_cifar10(data_dir="data"):
    """CIFAR-10: train_x (50000, 3, 32, 32) uint8, train_y, test_x (10000, ...),
    test_y; classes in CIFAR10_CLASSES order."""

    def build():
        _fetch_tgz(FASTAI + "cifar10.tgz", data_dir, "cifar10")
        out = {}
        for split in ("train", "test"):
            xs, ys = [], []
            for label, name in enumerate(CIFAR10_CLASSES):
                folder = os.path.join(data_dir, "cifar10", split, name)
                for f in sorted(os.listdir(folder)):
                    with open(os.path.join(folder, f), "rb") as fh:
                        xs.append(_image(fh.read(), 32))
                    ys.append(label)
            out[f"{split}_x"], out[f"{split}_y"] = np.stack(xs), np.asarray(ys, np.int64)
        return out

    return _cached(os.path.join(data_dir, "cifar10.npz"), build)


# ------------------------------------------------------------ Oxford-IIIT Pet

def load_pets(data_dir="data", size=64):
    """Oxford-IIIT Pet at size x size, official trainval / test split.

    {split}_x: (N, 3, size, size) uint8 images; {split}_mask: (N, size, size)
    uint8 trimap classes 0 pet, 1 background, 2 border (PET_MASK_CLASSES);
    {split}_species: 0 cat, 1 dog; {split}_breed: 0..36; {split}_box: (N, 4)
    float32 head box (x0, y0, x1, y1) in [0, 1], NaN where the dataset has
    none (the head boxes cover most of trainval and none of test)."""

    def build():
        import xml.etree.ElementTree as ET

        _fetch_tgz(FASTAI + "oxford-iiit-pet.tgz", data_dir, "oxford-iiit-pet")
        root = os.path.join(data_dir, "oxford-iiit-pet")
        out = {}
        for split, listing in (("train", "trainval.txt"), ("test", "test.txt")):
            xs, ms, sp, br, bx = [], [], [], [], []
            with open(os.path.join(root, "annotations", listing)) as f:
                rows = [line.split() for line in f if line.strip() and not line.startswith("#")]
            for name, cls, species, _ in rows:
                img = os.path.join(root, "images", name + ".jpg")
                if not os.path.exists(img):
                    continue
                xs.append(_image(img, size))
                tri = _image(os.path.join(root, "annotations", "trimaps", name + ".png"), size, "L")
                ms.append((tri.astype(np.int64) - 1).clip(0, 2).astype(np.uint8))  # 1,2,3 -> 0,1,2
                sp.append(int(species) - 1)
                br.append(int(cls) - 1)
                box = [np.nan] * 4
                xml = os.path.join(root, "annotations", "xmls", name + ".xml")
                if os.path.exists(xml):
                    a = ET.parse(xml).getroot()
                    w, h = float(a.find("size/width").text), float(a.find("size/height").text)
                    b = a.find("object/bndbox")
                    box = [float(b.find("xmin").text) / w, float(b.find("ymin").text) / h,
                           float(b.find("xmax").text) / w, float(b.find("ymax").text) / h]
                bx.append(box)
            out[f"{split}_x"], out[f"{split}_mask"] = np.stack(xs), np.stack(ms)
            out[f"{split}_species"], out[f"{split}_breed"] = np.asarray(sp, np.int64), np.asarray(br, np.int64)
            out[f"{split}_box"] = np.asarray(bx, np.float32)
        return out

    return _cached(os.path.join(data_dir, f"pets{size}.npz"), build)


# ----------------------------------------------------------------- COCO 2017

class _RemoteZip:
    """Random access to the members of a zip file on an HTTP server that
    serves byte ranges: reads the central directory once, then fetches each
    member with one range request (train2017.zip is 19 GB; an example needs
    a few thousand of its images)."""

    def __init__(self, url):
        self.url = url
        self.size = None
        self._get(0, 0)  # learns the size from Content-Range
        tail = self._get(max(0, self.size - (1 << 16)), self.size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64
            loc = tail.rfind(b"PK\x06\x07")
            (z64_off,) = struct.unpack("<Q", tail[loc + 8:loc + 16])
            rec = self._get(z64_off, z64_off + 55)
            cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
        cd = self._get(cd_off, cd_off + cd_size - 1)
        self.members = {}
        p = 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
            (off,) = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode()
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            vals = [v for v, big in ((usize, usize == 0xFFFFFFFF), (csize, csize == 0xFFFFFFFF),
                                     (off, off == 0xFFFFFFFF)) if big]
            if vals:  # zip64 extra field: the oversized values in this order
                q = 0
                while q + 4 <= len(extra):
                    tag, ln = struct.unpack("<HH", extra[q:q + 4])
                    if tag == 1:
                        big = struct.unpack("<" + "Q" * (ln // 8), extra[q + 4:q + 4 + ln])
                        it = iter(big)
                        if usize == 0xFFFFFFFF:
                            usize = next(it)
                        if csize == 0xFFFFFFFF:
                            csize = next(it)
                        if off == 0xFFFFFFFF:
                            off = next(it)
                    q += 4 + ln
            self.members[name] = (off, csize, method)
            p += 46 + nlen + elen + clen

    def _get(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if self.size is None:
                        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                    return r.read()
            except (OSError, http.client.HTTPException):  # resets, 502s, truncated bodies
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def read(self, name):
        off, csize, method = self.members[name]
        raw = self._get(off, off + 30 + 1024 + csize)
        nlen, elen = struct.unpack("<HH", raw[26:30])
        data = raw[30 + nlen + elen:30 + nlen + elen + csize]
        return zlib.decompress(data, -15) if method == 8 else data


def _rasterize(seg, h, w, size):
    """COCO segmentation (polygons or uncompressed RLE) -> (size, size) bool."""
    from PIL import Image, ImageDraw

    if isinstance(seg, list):
        im = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(im)
        for poly in seg:
            if len(poly) >= 6:
                d.polygon([(x * size / w, y * size / h) for x, y in zip(poly[::2], poly[1::2])], fill=1)
        return np.asarray(im, dtype=bool)
    counts = seg["counts"]
    if isinstance(counts, str):  # compressed RLE (not used in the instance files' crowds)
        return np.zeros((size, size), bool)
    flat = np.zeros(h * w, bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    full = flat.reshape(w, h).T  # column-major
    yi = (np.arange(size) * h / size).astype(int)
    xi = (np.arange(size) * w / size).astype(int)
    return full[np.ix_(yi, xi)]


def load_coco(data_dir="data", split="val", size=64, max_images=None, masks=False, workers=16):
    """COCO 2017 instances, every image squashed to size x size.

    split "val" (5000 images, downloaded whole, 0.8 GB) or "train" (118k
    images; the first max_images in id order are fetched one by one from the
    19 GB archive). Returns x (N, 3, size, size) uint8, image_id (N,),
    box (M, 4) float32 (x0, y0, x1, y1) in [0, 1], box_image (M,) index into
    x, box_label (M,) 0..79, box_area (M,) in original pixels, box_crowd
    (M,), names (80,) category names; with masks, mask (N, size, size) uint8
    semantic labels (0 background, 1 + label, overlaps: the later
    annotation wins)."""
    tag = f"coco_{split}{size}" + (f"_{max_images}" if max_images else "") + ("_masks" if masks else "")

    def build():
        ann_zip = _download(COCO + "annotations_trainval2017.zip", os.path.join(data_dir, "annotations_trainval2017.zip"))
        with zipfile.ZipFile(ann_zip) as z:
            ann = json.loads(z.read(f"annotations/instances_{split}2017.json"))
        cats = sorted(ann["categories"], key=lambda c: c["id"])
        cat_index = {c["id"]: i for i, c in enumerate(cats)}
        images = sorted(ann["images"], key=lambda im: im["id"])
        if max_images:
            images = images[:max_images]
        if split == "val":
            img_zip = zipfile.ZipFile(_download(COCO + "val2017.zip", os.path.join(data_dir, "val2017.zip")))
            fetch = lambda name: img_zip.read(name)  # noqa: E731
            workers_ = 1
        else:
            remote = _RemoteZip(COCO + "train2017.zip")
            fetch = remote.read
            workers_ = workers
        by_image = {}
        for a in ann["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)

        def one(im):
            return _image(fetch(f"{split}2017/{im['file_name']}"), size)

        with concurrent.futures.ThreadPoolExecutor(workers_) as ex:
            xs = list(ex.map(one, images))
        boxes, bimg, blab, barea, bcrowd, ms = [], [], [], [], [], []
        for i, im in enumerate(images):
            w, h = float(im["width"]), float(im["height"])
            m = np.zeros((size, size), np.uint8)
            for a in by_image.get(im["id"], []):
                x, y, bw, bh = a["bbox"]
                boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
                bimg.append(i)
                blab.append(cat_index[a["category_id"]])
                barea.append(a["area"])
                bcrowd.append(a["iscrowd"])
                if masks:
                    m[_rasterize(a["segmentation"], int(h), int(w), size)] = 1 + cat_index[a["category_id"]]
            ms.append(m)
        out = dict(x=np.stack(xs), image_id=np.asarray([im["id"] for im in images], np.int64),
                   box=np.asarray(boxes, np.float32).reshape(-1, 4), box_image=np.asarray(bimg, np.int64),
                   box_label=np.asarray(blab, np.int64), box_area=np.asarray(barea, np.float32),
                   box_crowd=np.asarray(bcrowd, np.uint8), names=np.asarray([c["name"] for c in cats]))
        if masks:
            out["mask"] = np.stack(ms)
        return out

    return _cached(os.path.join(data_dir, tag + ".npz"), build)


# ------------------------------------------------------------------- CMUdict

def load_cmudict(data_dir="data", max_letters=8, max_phonemes=8):
    """Grapheme-to-phoneme pairs from the CMU pronouncing dictionary.

    Words of a-z letters only, first pronunciation, stress marks removed (39
    phonemes); at most max_letters letters and max_phonemes phonemes. Split
    by a hash of the word: about 10% test. Returns letters (27: pad + a-z),
    phonemes (42: pad, <s>, </s>, then the 39), and per split src (N,
    max_letters) and tgt (N, max_phonemes + 1) int64: src is letter ids
    padded with 0; tgt is the phoneme ids followed by </s>, padded with 0.
    Decoders take <s> + tgt[:-1] as input (teacher forcing)."""

    def build():
        path = _download(CMUDICT, os.path.join(data_dir, "cmudict.dict"))
        letters = ["<pad>"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        pairs, phones = [], set()
        with open(path, encoding="latin-1") as f:
            for line in f:
                parts = line.split("#")[0].split()
                if len(parts) < 2 or "(" in parts[0] or not parts[0].isalpha() or not parts[0].isascii():
                    continue
                word = parts[0].lower()
                ph = ["".join(c for c in p if not c.isdigit()) for p in parts[1:]]
                if len(word) <= max_letters and len(ph) <= max_phonemes:
                    pairs.append((word, ph))
                    phones.update(ph)
        phonemes = ["<pad>", "<s>", "</s>"] + sorted(phones)
        li, pi = {c: i for i, c in enumerate(letters)}, {p: i for i, p in enumerate(phonemes)}
        out = dict(letters=np.asarray(letters), phonemes=np.asarray(phonemes))
        test = [int(hashlib.md5(w.encode()).hexdigest(), 16) % 10 == 0 for w, _ in pairs]
        for split, keep in (("train", False), ("test", True)):
            sel = [p for p, t in zip(pairs, test) if t == keep]
            src = np.zeros((len(sel), max_letters), np.int64)
            tgt = np.zeros((len(sel), max_phonemes + 1), np.int64)
            for i, (w, ph) in enumerate(sel):
                src[i, :len(w)] = [li[c] for c in w]
                tgt[i, :len(ph) + 1] = [pi[p] for p in ph] + [pi["</s>"]]
            out[f"{split}_src"], out[f"{split}_tgt"] = src, tgt
            out[f"{split}_words"] = np.asarray([w for w, _ in sel])
        return out

    return _cached(os.path.join(data_dir, f"cmudict_{max_letters}_{max_phonemes}.npz"), build)


# ---------------------------------------------------------------------- text

_WORD = None


def tokenize(text):
    """Lower-cased words, numbers and single punctuation marks."""
    global _WORD
    if _WORD is None:
        import re

        _WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
    return _WORD.findall(text.lower().replace("\\", " "))


def _encode_text(train, test, max_tokens, vocab_size):
    """Shared by the text loaders: train/test are lists of (words, label).
    The vocabulary is the vocab_size - 2 most frequent training words after
    <pad> (0) and <unk> (1); every text is cut to its first max_tokens
    words and padded with 0. Returns the dict of arrays."""
    from collections import Counter

    counts = Counter(w for words, _ in train for w in words)
    vocab = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(vocab_size - 2)]
    index = {w: i for i, w in enumerate(vocab)}
    out = dict(vocab=np.asarray(vocab))
    for split, rows in (("train", train), ("test", test)):
        x = np.zeros((len(rows), max_tokens), np.int64)
        for i, (words, _) in enumerate(rows):
            ids = [index.get(w, 1) for w in words[:max_tokens]]
            x[i, :len(ids)] = ids
        out[f"{split}_x"] = x
        out[f"{split}_y"] = np.asarray([y for _, y in rows], np.int64)
        out[f"{split}_len"] = np.asarray([min(len(w), max_tokens) for w, _ in rows], np.int64)
    return out


def load_agnews(data_dir="data", max_tokens=16, vocab_size=20000):
    """AG News topic classification (Zhang et al. 2015): 120,000 training and
    7,600 test articles, 4 balanced classes (AGNEWS_CLASSES). The text is the
    title followed by the description, tokenized by ``tokenize`` and cut to
    max_tokens words. Returns vocab and train/test x (N, max_tokens) int64
    word ids (0 = pad, 1 = unknown), y (N,) and len (N,) (words kept)."""

    def build():
        import csv

        _fetch_tgz(AGNEWS, data_dir, "ag_news_csv")
        splits = []
        for split in ("train", "test"):
            with open(os.path.join(data_dir, "ag_news_csv", split + ".csv"), encoding="utf-8") as f:
                rows = csv.reader(f)
                splits.append([(tokenize(title + " . " + desc), int(label) - 1) for label, title, desc in rows])
        return _encode_text(*splits, max_tokens, vocab_size)

    return _cached(os.path.join(data_dir, f"agnews_{max_tokens}_{vocab_size}.npz"), build)


def load_sst2(data_dir="data", max_tokens=32, vocab_size=12000):
    """SST-2 binary sentiment (Socher et al. 2013), sentence level: 6,920
    training, 872 dev and 1,821 test sentences (GLUE's test labels are
    hidden; this split has them). Same arrays as load_agnews; the train
    split here includes the dev sentences."""

    def build():
        splits = []
        for names in (("train", "dev"), ("test",)):
            rows = []
            for name in names:
                path = _download(SST2 + name + ".tsv", os.path.join(data_dir, "sst2", name + ".tsv"))
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        text, label = line.rstrip("\n").rsplit("\t", 1)
                        rows.append((tokenize(text), int(label)))
            splits.append(rows)
        return _encode_text(*splits, max_tokens, vocab_size)

    return _cached(os.path.join(data_dir, f"sst2_{max_tokens}_{vocab_size}.npz"), build)


LOADERS = dict(mnist=load_mnist, cifar10=load_cifar10, pets=load_pets, coco=load_coco, cmudict=load_cmudict,
               agnews=load_agnews, sst2=load_sst2)


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(LOADERS))
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    for name in args.names:
        d = LOADERS[name](args.data)
        print(name + ": " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in d.items()))


if __name__ == "__main__":
    main()
