#!/usr/bin/env python3
"""Build player headshot cutouts from Wikimedia Commons."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from rembg import new_session, remove
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("headshots")

USER_AGENT = "tennis-wire-headshots/0.1 (https://github.com/tennis-wire/tennis-wire)"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikidata entities used to reject same-name non-players.
OCCUPATION_TENNIS_PLAYER = "Q10833314"  # P106
SPORT_TENNIS = "Q847"  # P641

# The YuNet weights are stored via Git LFS. raw.githubusercontent.com serves a
# 131-byte pointer file, not the model; only the media host returns real weights.
YUNET_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
YUNET_MIN_BYTES = 100_000

# CC BY requires stating that the work was modified.
MODIFICATIONS = ("cropped", "background_removed")

DEFAULT_MODEL = "birefnet-general"
COMPARE_MODELS = ("birefnet-general", "birefnet-portrait", "isnet-general-use", "u2net_human_seg")


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Config:
    names: Path
    out: Path
    model: str
    side: int
    detect_max: int
    min_face_px: int
    face_scale: float
    top_pad: float
    score: float
    jpeg_quality: int
    force: bool

    @property
    def raw_dir(self) -> Path:
        return self.out / "raw"

    @property
    def done_dir(self) -> Path:
        return self.out / "headshots"

    @property
    def compare_dir(self) -> Path:
        return self.out / "compare"

    @property
    def yunet(self) -> Path:
        return self.out / "yunet.onnx"


@dataclass
class Photo:
    """One produced headshot, with everything needed to attribute it."""

    player: str
    qid: str
    cutout: str
    crop: str
    source_url: str
    author: str
    license: str
    license_url: str
    modifications: list[str]
    restrictions: str
    share_alike: bool
    rembg_model: str
    alpha_coverage: float
    is_derivative: bool = True
    warnings: list[str] = field(default_factory=list)


class SkipReason(Exception):
    """A player was skipped for an expected, reportable reason."""


@dataclass(frozen=True)
class Target:
    """One player to process. A qid from roster.py skips the name search."""

    name: str
    qid: str = ""


# --------------------------------------------------------------------------- http


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP = build_session()


def api_get(url: str, **params: object) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    response = HTTP.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def ensure_yunet(path: Path) -> None:
    """Download the face detector, rejecting Git LFS pointer files."""
    if path.exists() and path.stat().st_size >= YUNET_MIN_BYTES:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in YUNET_URLS:
        try:
            log.info("downloading face detector from %s", url.split("/")[2])
            response = HTTP.get(url, timeout=180)
            response.raise_for_status()
            if len(response.content) < YUNET_MIN_BYTES:
                log.warning("got %d bytes (LFS pointer), trying next source", len(response.content))
                continue
            path.write_bytes(response.content)
            log.info("saved %d bytes to %s", len(response.content), path)
            return
        except requests.RequestException as exc:
            log.warning("download failed: %s", exc)
    raise SystemExit(
        f"could not download the face detector.\n"
        f"Fetch it manually from {YUNET_URLS[0]} and save it as {path} "
        f"(expected ~232 KB, not 131 bytes)."
    )


# --------------------------------------------------------------------------- metadata


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


# Letters with strokes survive NFKD decomposition and would otherwise be dropped.
# All of these occur in tennis rosters.
TRANSLITERATE = str.maketrans(
    {
        "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
        "Ł": "L", "ł": "l", "Ø": "O", "ø": "o",
        "Þ": "Th", "þ": "th", "ß": "ss",
        "Æ": "Ae", "æ": "ae", "Œ": "Oe", "œ": "oe",
        "İ": "I", "ı": "i", "Ħ": "H", "ħ": "h",
    }
)


def slugify(name: str) -> str:
    """ASCII-only slug. Tennis rosters are full of diacritics and these end up
    as object keys, so fold them rather than passing them through. Names that
    fold to nothing (CJK, Cyrillic) get a hashed slug so they stay unique."""
    folded = unicodedata.normalize("NFKD", name.translate(TRANSLITERATE))
    ascii_only = folded.encode("ascii", "ignore").decode()
    joined = "".join(char if char.isalnum() else "_" for char in ascii_only.lower())
    slug = re.sub(r"_+", "_", joined).strip("_")
    if not slug:
        return "player_" + hashlib.sha1(name.encode()).hexdigest()[:10]
    return slug


def search_entities(name: str) -> list[str]:
    hits = api_get(
        WIKIDATA_API,
        action="wbsearchentities",
        language="en",
        uselang="en",
        type="item",
        limit=7,
        search=name,
    ).get("search", [])
    return [hit["id"] for hit in hits]


def _entity_ids(claims: dict, prop: str) -> list[str]:
    ids = []
    for claim in claims.get(prop, []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            ids.append(value["id"])
    return ids


def resolve_player(name: str, qid: str = "") -> tuple[str, str]:
    """Return (qid, commons_filename). With a known qid the name search is
    skipped entirely, which removes the same-name failure mode. Raises
    SkipReason otherwise."""
    candidates = [qid] if qid else search_entities(name)
    if not candidates:
        raise SkipReason("not found in Wikidata")

    entities = api_get(
        WIKIDATA_API,
        action="wbgetentities",
        ids="|".join(candidates),
        props="claims",
    ).get("entities", {})

    saw_player = False
    for candidate in candidates:  # preserve search ranking
        claims = entities.get(candidate, {}).get("claims", {})
        if not claims:
            continue
        if not qid:  # a caller-supplied qid is trusted; a searched one is not
            is_player = OCCUPATION_TENNIS_PLAYER in _entity_ids(
                claims, "P106"
            ) or SPORT_TENNIS in _entity_ids(claims, "P641")
            if not is_player:
                continue
        saw_player = True
        for claim in claims.get("P18", []):
            filename = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(filename, str):
                return candidate, filename

    raise SkipReason("no image (P18)" if saw_player else "no tennis player matched")


def commons_metadata(filename: str) -> dict[str, str]:
    pages = (
        api_get(
            COMMONS_API,
            action="query",
            prop="imageinfo",
            iiprop="url|extmetadata|size",
            titles=f"File:{filename}",
        )
        .get("query", {})
        .get("pages", [])
    )
    info = (pages[0].get("imageinfo") if pages else None) or [None]
    if not info[0]:
        raise SkipReason("file not readable on Commons")

    meta = info[0].get("extmetadata", {})

    def field_value(key: str) -> str:
        return strip_html(meta.get(key, {}).get("value", "") or "")

    author = field_value("Artist")
    licence = field_value("LicenseShortName")
    if not author or not licence:
        raise SkipReason("missing author or licence")

    return {
        "url": info[0]["url"],
        "descriptionurl": info[0]["descriptionurl"],
        "author": author,
        "license": licence,
        "license_url": field_value("LicenseUrl"),
        "restrictions": field_value("Restrictions"),
    }


def is_share_alike(licence: str) -> bool:
    return "SA" in licence.upper().replace("-", "").replace(" ", "")


# --------------------------------------------------------------------------- imaging


def detect_face(image: np.ndarray, cfg: Config) -> tuple[float, float, float, float]:
    """Largest face, in original-image coordinates. Detection runs on a
    downscaled copy because YuNet degrades on very large inputs."""
    height, width = image.shape[:2]
    scale = min(1.0, cfg.detect_max / max(height, width))
    small = (
        cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image
    )
    small_h, small_w = small.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        str(cfg.yunet), "", (small_w, small_h), score_threshold=cfg.score
    )
    detector.setInputSize((small_w, small_h))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        raise SkipReason("no face detected")

    box = faces[np.argmax(faces[:, 2] * faces[:, 3])][:4]
    face_x, face_y, face_w, face_h = (float(v) / scale for v in box)
    if face_w < cfg.min_face_px:
        raise SkipReason(f"face too small ({int(face_w)}px < {cfg.min_face_px})")
    return face_x, face_y, face_w, face_h


def crop_head(
    image: np.ndarray, box: tuple[float, float, float, float], cfg: Config
) -> np.ndarray:
    """Square crop around head and shoulders, clamped to the frame."""
    height, width = image.shape[:2]
    face_x, face_y, face_w, face_h = box

    side = min(face_w * cfg.face_scale, float(width), float(height))
    left = max(0.0, min(face_x + face_w / 2 - side / 2, width - side))
    top = max(0.0, min(face_y - face_h * cfg.top_pad, height - side))

    x, y, size = round(left), round(top), round(side)
    patch = image[y : y + size, x : x + size]
    return cv2.resize(patch, (cfg.side, cfg.side), interpolation=cv2.INTER_AREA)


def alpha_coverage(image: Image.Image) -> float:
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    return float((alpha > 16).mean())


def load_crop(target: Target, cfg: Config) -> tuple[Image.Image, dict[str, str], str]:
    """Fetch the source image and return its head crop plus licence metadata."""
    qid, filename = resolve_player(target.name, target.qid)
    meta = commons_metadata(filename)

    raw_path = cfg.raw_dir / f"{slugify(target.name)}.img"
    if cfg.force or not raw_path.exists():
        response = HTTP.get(meta["url"], timeout=120)
        response.raise_for_status()
        raw_path.write_bytes(response.content)

    image = cv2.imread(str(raw_path))
    if image is None:
        raise SkipReason("could not decode image")

    crop = crop_head(image, detect_face(image, cfg), cfg)
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), meta, qid


# --------------------------------------------------------------------------- reports

CSS = """
body{background:#15161a;color:#e8e8ea;font:13px/1.45 system-ui,sans-serif;margin:24px}
h1{font-size:15px;font-weight:600;margin:0 0 4px}
p.sub{color:#8a8b92;margin:0 0 20px}
.grid{display:flex;flex-wrap:wrap;gap:20px}
figure{margin:0;width:210px}
.pair{display:flex;gap:4px}
.ph{border-radius:6px;overflow:hidden;width:103px;height:103px;
 background-image:linear-gradient(45deg,#33343a 25%,transparent 25%),
  linear-gradient(-45deg,#33343a 25%,transparent 25%),
  linear-gradient(45deg,transparent 75%,#33343a 75%),
  linear-gradient(-45deg,transparent 75%,#33343a 75%);
 background-size:14px 14px;background-position:0 0,0 7px,7px -7px,-7px 0}
.ph.wide{width:200px;height:200px}
.ph img{width:100%;height:100%;object-fit:contain;display:block}
figcaption{margin-top:7px;color:#a6a7ad;word-break:break-word}
figcaption b{color:#e8e8ea}
.warn{color:#e0a33c}
.sa{color:#6ea8fe}
.row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:26px}
.row h2{font-size:13px;width:100%;margin:0 0 6px;color:#e8e8ea}
.cell{width:200px;text-align:center;color:#8a8b92;font-size:11px}
"""


def write_review(photos: list[Photo], cfg: Config) -> None:
    cards = []
    for photo in photos:
        notes = ""
        if photo.share_alike:
            notes += '<br><span class="sa">cutout stays BY-SA</span>'
        if photo.warnings:
            notes += f'<br><span class="warn">{"; ".join(photo.warnings)}</span>'
        cards.append(
            f'<figure><div class="pair">'
            f'<div class="ph"><img src="headshots/{photo.cutout}"></div>'
            f'<div class="ph"><img src="headshots/{photo.crop}"></div></div>'
            f"<figcaption><b>{photo.player}</b><br><small>{photo.author}<br>"
            f"{photo.license} &middot; alpha {photo.alpha_coverage:.0%}"
            f"{notes}</small></figcaption></figure>"
        )
    html = (
        f"<!doctype html><meta charset=utf-8><title>headshots</title><style>{CSS}</style>"
        f"<h1>{len(photos)} results &mdash; cutout left, cropped source right</h1>"
        f'<p class="sub">Model: {cfg.model}. The checkerboard exposes gaps in the '
        f"alpha channel; check hair edges and the shoulder line.</p>"
        f'<div class="grid">{"".join(cards)}</div>'
    )
    (cfg.out / "review.html").write_text(html, encoding="utf-8")


def write_comparison(rows: list[tuple[str, list[tuple[str, str, float]]]], cfg: Config) -> None:
    body = []
    for player, cells in rows:
        rendered = "".join(
            f'<div class="cell"><div class="ph wide"><img src="compare/{filename}"></div>'
            f'<b style="color:#e8e8ea">{model}</b><br>alpha {coverage:.0%}</div>'
            for model, filename, coverage in cells
        )
        body.append(f'<div class="row"><h2>{player}</h2>{rendered}</div>')
    html = (
        f"<!doctype html><meta charset=utf-8><title>model comparison</title>"
        f"<style>{CSS}</style><h1>Cutout model comparison</h1>"
        f'<p class="sub">Same crop through each model. Pass the winner via --model.</p>'
        f'{"".join(body)}'
    )
    (cfg.out / "compare.html").write_text(html, encoding="utf-8")


# --------------------------------------------------------------------------- commands


def read_targets(path: Path) -> list[Target]:
    """Accept either a plain name-per-line file or the CSV roster.py writes."""
    if not path.exists():
        raise SystemExit(f"{path} not found")

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "name" not in rows[0]:
            raise SystemExit(f"{path} needs at least a 'name' column")
        targets = [
            Target(name=row["name"].strip(), qid=(row.get("qid") or "").strip())
            for row in rows
            if row.get("name", "").strip()
        ]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        targets = [
            Target(name=line.strip())
            for line in lines
            if line.strip() and not line.startswith("#")
        ]

    if not targets:
        raise SystemExit(f"{path} contains no names")
    return targets


def build(targets: list[Target], cfg: Config) -> int:
    log.info("rembg model: %s", cfg.model)
    session = new_session(cfg.model)

    photos: list[Photo] = []
    skipped: Counter[str] = Counter()

    for index, target in enumerate(targets, 1):
        name = target.name
        log.info("[%d/%d] %s", index, len(targets), name)
        try:
            crop, meta, qid = load_crop(target, cfg)
        except SkipReason as reason:
            skipped[str(reason)] += 1
            log.warning("  skipped: %s", reason)
            continue
        except requests.RequestException as exc:
            skipped[f"http error: {type(exc).__name__}"] += 1
            log.warning("  skipped: %s", exc)
            continue

        cutout = remove(crop, session=session).convert("RGBA")
        coverage = alpha_coverage(cutout)

        warnings = []
        if coverage < 0.15:
            warnings.append("almost everything removed")
        elif coverage > 0.97:
            warnings.append("background not separated")
        if meta["restrictions"]:
            warnings.append(meta["restrictions"][:60])

        slug = slugify(name)
        photo = Photo(
            player=name,
            qid=qid,
            cutout=f"{slug}.png",
            crop=f"{slug}.jpg",
            source_url=meta["descriptionurl"],
            author=meta["author"],
            license=meta["license"],
            license_url=meta["license_url"],
            modifications=list(MODIFICATIONS),
            restrictions=meta["restrictions"],
            share_alike=is_share_alike(meta["license"]),
            rembg_model=cfg.model,
            alpha_coverage=coverage,
            warnings=warnings,
        )

        cutout.save(cfg.done_dir / photo.cutout)
        crop.save(cfg.done_dir / photo.crop, quality=cfg.jpeg_quality)
        photos.append(photo)

        log.info(
            "  ok: %s / %s, alpha %.0f%%%s",
            photo.license,
            photo.author[:40],
            coverage * 100,
            "  [" + "; ".join(warnings) + "]" if warnings else "",
        )

    manifest = cfg.done_dir / "manifest.json"
    manifest.write_text(
        json.dumps([asdict(photo) for photo in photos], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_review(photos, cfg)

    clean = [photo for photo in photos if not photo.warnings]
    share_alike = [photo for photo in photos if photo.share_alike]
    total = len(targets)
    log.info("%s", "-" * 52)
    log.info("produced      %d/%d (%.0f%%)", len(photos), total, len(photos) / total * 100)
    log.info("without flags %d/%d (%.0f%%)", len(clean), total, len(clean) / total * 100)
    log.info("BY-SA         %d (cutout inherits ShareAlike)", len(share_alike))
    for reason, count in skipped.most_common():
        log.info("  %3d  %s", count, reason)
    log.info("manifest: %s", manifest)
    log.info("review:   %s", cfg.out / "review.html")
    return 0 if photos else 1


def compare(targets: list[Target], models: list[str], cfg: Config) -> int:
    """Run the same crops through several models. Crops are computed once and
    models are loaded one at a time to keep memory flat."""
    cfg.compare_dir.mkdir(parents=True, exist_ok=True)

    crops: list[tuple[str, Image.Image]] = []
    for index, target in enumerate(targets, 1):
        name = target.name
        log.info("[%d/%d] %s", index, len(targets), name)
        try:
            crop, _, _ = load_crop(target, cfg)
        except (SkipReason, requests.RequestException) as reason:
            log.warning("  skipped: %s", reason)
            continue
        crops.append((name, crop))

    if not crops:
        log.error("nothing to compare")
        return 1

    results: dict[str, list[tuple[str, str, float]]] = {name: [] for name, _ in crops}
    for model in models:
        log.info("=== %s", model)
        try:
            session = new_session(model)
        except Exception as exc:  # noqa: BLE001 - model names come from the CLI
            log.warning("  unavailable: %s", exc)
            continue
        for name, crop in crops:
            cutout = remove(crop, session=session).convert("RGBA")
            filename = f"{slugify(name)}__{model}.png"
            cutout.save(cfg.compare_dir / filename)
            coverage = alpha_coverage(cutout)
            results[name].append((model, filename, coverage))
            log.info("  %-28s alpha %.0f%%", name, coverage * 100)
        del session
        gc.collect()

    write_comparison([(name, results[name]) for name, _ in crops], cfg)
    log.info("comparison: %s", cfg.out / "compare.html")
    return 0


# --------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="headshots",
        description="Build player headshot cutouts from Wikimedia Commons.",
    )
    parser.add_argument("names", type=Path, nargs="?", default=Path("players.txt"),
                        help="file with one player name per line (default: players.txt)")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"),
                        help="output directory (default: out)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"rembg model (default: {DEFAULT_MODEL})")
    parser.add_argument("--compare", nargs="*", metavar="MODEL", default=None,
                        help="compare models instead of building; defaults to a preset list")
    parser.add_argument("--side", type=int, default=800, help="output edge in px (default: 800)")
    parser.add_argument("--detect-max", type=int, default=1280,
                        help="downscale before face detection (default: 1280)")
    parser.add_argument("--min-face", type=int, default=140, dest="min_face_px",
                        help="reject sources whose face is narrower, in px (default: 140)")
    parser.add_argument("--face-scale", type=float, default=3.0,
                        help="crop width in face widths; lower is tighter (default: 3.0)")
    parser.add_argument("--top-pad", type=float, default=0.85,
                        help="headroom above the face, in face heights (default: 0.85)")
    parser.add_argument("--score", type=float, default=0.7,
                        help="face detector confidence threshold (default: 0.7)")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (default: 90)")
    parser.add_argument("--force", action="store_true", help="re-download cached sources")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    cfg = Config(
        names=args.names,
        out=args.out,
        model=args.model,
        side=args.side,
        detect_max=args.detect_max,
        min_face_px=args.min_face_px,
        face_scale=args.face_scale,
        top_pad=args.top_pad,
        score=args.score,
        jpeg_quality=args.jpeg_quality,
        force=args.force,
    )

    targets = read_targets(cfg.names)
    for directory in (cfg.raw_dir, cfg.done_dir):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_yunet(cfg.yunet)

    if args.compare is not None:
        return compare(targets, args.compare or list(COMPARE_MODELS), cfg)
    return build(targets, cfg)


if __name__ == "__main__":
    raise SystemExit(main())