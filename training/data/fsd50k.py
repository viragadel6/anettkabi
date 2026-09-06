"""FSD50K dataset acquisition: Zenodo download, unzip, label parsing.

Verified against the live Zenodo record 4060432 ("FSD50K", CC-BY-4.0):

* dev audio ships as ``FSD50K.dev_audio.z01..z05`` + ``FSD50K.dev_audio.zip``
  (~18.4 GB total);
* eval audio ships as ``FSD50K.eval_audio.z01`` + ``FSD50K.eval_audio.zip``
  (~6.3 GB) on the SAME record (there is no separate eval record);
* ground truth is ``FSD50K.ground_truth.zip`` with ``dev.csv``/``eval.csv``
  (40,966 + 10,231 clips).

The archive file list is enumerated from the record API at runtime so part
counts can drift upstream without breaking this script; hardcoded fallbacks
encode the layout above.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx

from training.config import TrainSettings, get_train_settings

__all__ = [
    "clip_paths",
    "download_fsd50k",
    "prepare_fsd50k_index",
    "record_files",
    "select_archive_names",
]

GROUND_TRUTH = "FSD50K.ground_truth.zip"
DEV_FALLBACK = [f"FSD50K.dev_audio.z{index:02d}" for index in range(1, 6)] + ["FSD50K.dev_audio.zip"]
EVAL_FALLBACK = ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"]


def _record_id(base_url: str) -> str:
    """Extract the numeric Zenodo record id from a record URL.

    Parameters:
        base_url: Record URL (e.g. https://zenodo.org/records/4060432).

    Returns:
        The record id string.
    """
    return base_url.rstrip("/").rsplit("/", 1)[-1]


def record_files(base_url: str) -> list[tuple[str, int]] | None:
    """List (filename, size) pairs of a Zenodo record via its API.

    Parameters:
        base_url: Record URL.

    Returns:
        Sorted file list, or None when the API is unreachable (callers fall
        back to the hardcoded layout).
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            response = client.get(f"https://zenodo.org/api/records/{_record_id(base_url)}")
            response.raise_for_status()
            entries = response.json().get("files", [])
            return sorted((str(entry["key"]), int(entry["size"])) for entry in entries)
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def select_archive_names(settings: TrainSettings, include_eval: bool) -> list[str]:
    """Choose the archive file names to download.

    Preference order: live record API enumeration, then hardcoded fallbacks.

    Parameters:
        settings: Training settings (dataset URLs).
        include_eval: Whether eval audio is wanted.

    Returns:
        Ordered file-name list (dev parts, ground truth, then eval parts).
    """
    dev_remote = record_files(settings.dataset_url_fsd50k_dev)
    if dev_remote is not None:
        names = sorted(key for key, _size in dev_remote if key.startswith("FSD50K.dev_audio."))
        ground_truth = [key for key, _size in dev_remote if key == GROUND_TRUTH]
    else:
        names = list(DEV_FALLBACK)
        ground_truth = [GROUND_TRUTH]
    selected = [*names, *ground_truth]
    if not include_eval:
        return selected
    eval_remote = record_files(settings.dataset_url_fsd50k_eval) or dev_remote
    if eval_remote is not None:
        eval_names = sorted(key for key, _size in eval_remote if key.startswith("FSD50K.eval_audio."))
    else:
        eval_names = list(EVAL_FALLBACK)
    return [*selected, *eval_names]


def _zenodo_files(base_url: str, names: list[str]) -> list[str]:
    """Build direct Zenodo file URLs.

    Parameters:
        base_url: Record URL.
        names: File names.

    Returns:
        File URL list.
    """
    return [
        f"https://zenodo.org/records/{_record_id(base_url)}/files/{name}?download=1"
        for name in names
    ]


def download_fsd50k(settings: TrainSettings, include_eval: bool = True) -> Path:
    """Download and extract FSD50K into data_dir/fsd50k.

    Parameters:
        settings: Training settings.
        include_eval: Also fetch the eval split.

    Returns:
        The dataset root directory.

    Raises:
        RuntimeError: On download or extraction failure.
    """
    root = settings.data_dir / "fsd50k"
    audio_root = root / "FSD50K.dev_audio"
    if audio_root.is_dir() and any(audio_root.glob("*.wav")):
        print(f"FSD50K dev already present at {audio_root}")
        return root
    if (root / "dev").is_dir() and any((root / "dev").glob("*.wav")):
        print(f"FSD50K dev already present at {root / 'dev'}")
        return root
    root.mkdir(parents=True, exist_ok=True)
    archive_dir = root / "archives"
    archive_dir.mkdir(exist_ok=True)
    names = select_archive_names(settings, include_eval)
    urls_by_name: dict[str, str] = {}
    for name in names:
        record_url = (
            settings.dataset_url_fsd50k_eval
            if name.startswith("FSD50K.eval_audio.")
            else settings.dataset_url_fsd50k_dev
        )
        urls_by_name[name] = _zenodo_files(record_url, [name])[0]
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(3600.0)) as client:
        for name, url in urls_by_name.items():
            target = archive_dir / name
            if target.is_file() and target.stat().st_size > 0:
                print(f"skip existing {name}")
                continue
            print(f"downloading {name} ...")
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes(1024 * 1024):
                        handle.write(chunk)
    for zip_name in ("FSD50K.dev_audio.zip", "FSD50K.eval_audio.zip", GROUND_TRUTH):
        main_zip = archive_dir / zip_name
        if not main_zip.is_file():
            continue
        stem = zip_name[:-4]
        parts = [main_zip, *sorted(archive_dir.glob(stem + ".z*"))]
        out_dir = root / stem
        if not out_dir.is_dir() or not any(out_dir.iterdir()):
            out_dir.mkdir(exist_ok=True)
            _extract_split_zip(parts, out_dir)
    if any((root / "FSD50K.dev_audio").glob("*.wav")):
        (root / "FSD50K.dev_audio").rename(root / "dev")
    eval_dir = root / "FSD50K.eval_audio"
    if eval_dir.is_dir() and any(eval_dir.glob("*.wav")):
        eval_dir.rename(root / "eval")
    return root


def _extract_split_zip(parts: list[Path], out_dir: Path) -> None:
    """Extract a multi-part zip (split archives) via `zip -s 0` or 7z.

    Parameters:
        parts: [main.zip, .z01, ...] paths.
        out_dir: Extraction target.

    Raises:
        RuntimeError: When no extraction tool succeeds.
    """
    main_zip = parts[0]
    tools: list[list[str]] = []
    if shutil.which("zip"):
        tmp = out_dir.parent / (main_zip.stem + "_combined.zip")
        tools.append(["zip", "-s", "0", str(main_zip), "--out", str(tmp)])
        tools.append(["unzip", "-q", "-o", str(tmp), "-d", str(out_dir)])
    if shutil.which("7z"):
        tools.append(["7z", "x", "-y", f"-o{out_dir}", str(main_zip)])
    if not tools:
        raise RuntimeError("need `zip`/`unzip` or `7z` to extract split FSD50K archives")
    for command in tools:
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"extraction failed ({command[0]}): {result.stderr.decode()[:400]}"
            )


def prepare_fsd50k_index(settings: TrainSettings) -> Path:
    """Parse ground-truth CSVs into a clips index (id, labels, split, prompt).

    Parameters:
        settings: Training settings.

    Returns:
        Path of the written JSON index.

    Raises:
        FileNotFoundError: When ground truth is missing.
    """
    root = settings.data_dir / "fsd50k"
    gt_dir = root / "FSD50K.ground_truth"
    if not gt_dir.is_dir():
        extracted = list(root.glob("FSD50K.ground_truth*"))
        if not extracted:
            raise FileNotFoundError(f"FSD50K ground truth not found under {root}")
        gt_dir = extracted[0]
    index: list[dict[str, object]] = []
    csv_candidates = (
        ("dev.csv", ("FSD50K.ground_truth-dev.csv",), "dev"),
        ("eval.csv", ("FSD50K.ground_truth-eval.csv",), "eval"),
    )
    for primary_name, legacy_names, split in csv_candidates:
        csv_path = gt_dir / primary_name
        if not csv_path.is_file():
            for legacy in legacy_names:
                if (gt_dir / legacy).is_file():
                    csv_path = gt_dir / legacy
                    break
            else:
                continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                clip_id = str(row.get("fname") or row.get("clip_id") or "").strip()
                if not clip_id or clip_id == "fname":
                    continue
                labels = str(row.get("labels") or row.get("tags") or "").split(",")
                labels = [label.strip().replace("_", " ") for label in labels if label.strip()]
                entry: dict[str, object] = {
                    "clip_id": clip_id,
                    "split": split,
                    "labels": labels,
                    "prompt": ", ".join(labels[:6]),
                }
                subsplit = str(row.get("split") or "").strip()
                if subsplit:
                    entry["subsplit"] = subsplit
                index.append(entry)
    out = root / "index.json"
    out.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"index written: {out} ({len(index)} clips)")
    return out


def clip_paths(settings: TrainSettings, clip_id: str, split: str) -> Path:
    """Resolve one clip's wav path.

    Parameters:
        settings: Training settings.
        clip_id: Numeric clip id.
        split: dev|eval.

    Returns:
        Expected wav path.
    """
    root = settings.data_dir / "fsd50k"
    directory = root / split if (root / split).is_dir() else root / f"FSD50K.{split}_audio"
    return directory / f"{clip_id}.wav"


def iter_fsd50k(settings: TrainSettings) -> Iterator[dict[str, object]]:
    """Iterate indexed FSD50K clips with existing files.

    Parameters:
        settings: Training settings.

    Yields:
        Index entries whose wav exists.
    """
    root = settings.data_dir / "fsd50k"
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    for entry in index:
        if clip_paths(settings, str(entry["clip_id"]), str(entry["split"])).is_file():
            yield entry


def _main() -> int:
    """CLI entry for `python -m training.data.fsd50k --out DIR`.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/fsd50k")
    parser.add_argument("--dev-only", action="store_true", help="skip the eval audio archive")
    args = parser.parse_args()
    settings = get_train_settings()
    settings.data_dir = Path(args.out).parent
    download_fsd50k(settings, include_eval=not args.dev_only)
    prepare_fsd50k_index(settings)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
