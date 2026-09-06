"""VGGSound acquisition: metadata CSV + yt-dlp clip fetching for paired training."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx

from training.config import TrainSettings, get_train_settings

__all__ = ["download_vggsound_csv", "fetch_clip", "iter_vggsound", "prepare_vggsound"]

SEGMENT_S = 10.0
AUDIO_SR = 48000
SUPPORTED_LABEL_HINTS = (
    "typing",
    "footstep",
    "drum",
    "guitar",
    "dog",
    "car",
    "engine",
    "rain",
    "wind",
    "thunder",
    "door",
    "clapping",
    "chainsaw",
    "hammer",
    "jackhammer",
)


def download_vggsound_csv(settings: TrainSettings) -> Path:
    """Fetch the VGGSound metadata CSV.

    Parameters:
        settings: Training settings.

    Returns:
        Path of the CSV.

    Raises:
        RuntimeError: On download failure.
    """
    root = settings.data_dir / "vggsound"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "vggsound.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 10_000_000:
        return csv_path
    url = settings.dataset_vggsound_csv
    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
        response = client.get(url)
        response.raise_for_status()
        csv_path.write_bytes(response.content)
    return csv_path


def _label_ok(label: str) -> bool:
    """Filter labels to sound-effect-rich classes.

    Matching is word-prefix based (so `chainsaw` matches `chainsawing trees`,
    `hammer` matches `hammering`) while split markers like `train`/`test`
    never match the `rain` hint.

    Parameters:
        label: VGGSound class label.

    Returns:
        True when the label is SFX-relevant.
    """
    lowered = label.lower()
    words = re.split(r"[^a-z0-9]+", lowered)
    return any(word.startswith(hint) for word in words for hint in SUPPORTED_LABEL_HINTS)


def fetch_clip(youtube_id: str, start_s: float, destination: Path) -> bool:
    """Fetch one VGGSound segment via yt-dlp + ffmpeg trim.

    Parameters:
        youtube_id: YouTube video id.
        start_s: Segment start second.
        destination: Output audio path (.wav).

    Returns:
        True when the clip was fetched and trimmed.
    """
    if destination.is_file() and destination.stat().st_size > 10_000:
        return True
    raw = destination.with_suffix(".raw.m4a")
    command = [
        "yt-dlp",
        "-q",
        "--no-warnings",
        "--no-playlist",
        "-f",
        "bestaudio[ext=m4a]/bestaudio",
        "--download-sections",
        f"*{start_s:.0f}-{start_s + SEGMENT_S:.0f}",
        "--force-keyframes-at-cuts",
        "-o",
        str(raw),
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not raw.is_file():
        raw.unlink(missing_ok=True)
        return False
    trim = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw),
        "-t",
        f"{SEGMENT_S:.3f}",
        "-ac",
        "2",
        "-ar",
        str(AUDIO_SR),
        str(destination),
    ]
    trimmed = subprocess.run(trim, capture_output=True, check=False)
    raw.unlink(missing_ok=True)
    return trimmed.returncode == 0 and destination.is_file()


def prepare_vggsound(settings: TrainSettings, max_clips: int) -> Path:
    """Select, fetch, and index VGGSound audio clips.

    Parameters:
        settings: Training settings.
        max_clips: Maximum clips to fetch.

    Returns:
        Path of the written index JSON.

    Raises:
        FileNotFoundError: When the metadata CSV is missing.
    """
    root = settings.data_dir / "vggsound"
    csv_path = download_vggsound_csv(settings)
    candidates: list[dict[str, object]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            youtube_id, start, label = row[0], row[1], row[2]
            dataset_split = row[3].strip() if len(row) > 3 else ""
            try:
                start_s = float(start)
            except ValueError:
                continue
            if _label_ok(label):
                entry: dict[str, object] = {
                    "youtube_id": youtube_id,
                    "start_s": start_s,
                    "label": label,
                    "prompt": _prompt_for(label),
                }
                if dataset_split:
                    entry["dataset_split"] = dataset_split
                candidates.append(entry)
    rng = random.Random(settings.seed)
    rng.shuffle(candidates)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    fetched = 0
    for entry in candidates:
        if fetched >= max_clips:
            break
        destination = audio_dir / f"{entry['youtube_id']}_{int(entry['start_s']):06d}.wav"
        if fetch_clip(str(entry["youtube_id"]), float(entry["start_s"]), destination):
            entry["path"] = str(destination)
            index.append(entry)
            fetched += 1
            print(f"[{fetched}/{max_clips}] {destination.name}")
    out = root / "index.json"
    out.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"vggsound index: {out} ({len(index)} clips)")
    return out


def _prompt_for(label: str) -> str:
    """Convert a VGGSound class label into a sound-design prompt.

    Parameters:
        label: Raw class label like `people typing on keyboard`.

    Returns:
        Natural prompt string.
    """
    cleaned = label.strip().rstrip(".")
    return f"realistic {cleaned.replace('_', ' ')}, clean recording"


def iter_vggsound(settings: TrainSettings) -> Iterator[dict[str, object]]:
    """Iterate fetched VGGSound clips.

    Parameters:
        settings: Training settings.

    Yields:
        Index entries with existing audio.
    """
    index_path = settings.data_dir / "vggsound" / "index.json"
    if not index_path.is_file():
        return
    for entry in json.loads(index_path.read_text(encoding="utf-8")):
        if "path" in entry and Path(str(entry["path"])).is_file():
            yield entry


def _main() -> int:
    """CLI entry for `python -m training.data.vggsound --out DIR`.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/vggsound")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    settings = get_train_settings()
    settings.data_dir = Path(args.out).parent
    prepare_vggsound(settings, args.max or settings.vggsound_max_clips)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
