"""Generate real test media with FFmpeg lavfi/encoders for the test suite."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.media.ffmpeg import run_ffmpeg  # noqa: E402

SPECS: dict[str, list[str]] = {
    "standard.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        "-movflags", "+faststart",
    ],
    "silent_audio.mp4": [
        "-y", "-f", "lavfi", "-i", "smptebars=size=640x360:rate=25:duration=3",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
    ],
    "no_audio.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
    ],
    "moving_object.mp4": [
        "-y", "-f", "lavfi", "-i",
        "color=c=black:s=640x360:r=25,drawbox=x='mod(t*80,600)':y=150:w=40:h=40:color=red:t=fill",
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
        "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
    ],
    "noise_audio.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=3",
        "-f", "lavfi", "-i", "anoisesrc=color=brown:r=48000:a=0.5:d=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
    ],
    "vfr.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4",
        "-vf", "select='gt(t,0)+gt(t,1.4)+gt(t,2.2)+gt(t,3.1)'",
        "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000",
    ],
    "rotated.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=480x360:rate=25:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-metadata:s:v:0", "rotate=90",
    ],
    "clip_4k.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=3840x2160:rate=24:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
    ],
    "clip_0_4s.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=0.4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ],
    "clip_305s.mp4": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=305",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
    ],
    "webm_vp9.webm": [
        "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=2",
        "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
    ],
    "corrupt.mp4": [],
}


async def generate(out_dir: Path, only: list[str] | None) -> None:
    """Render the fixture media set.

    Parameters:
        out_dir: Output directory.
        only: Optional subset of names.

    Raises:
        FfmpegError: When an encoder run fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, args in SPECS.items():
        if only and name not in only:
            continue
        target = out_dir / name
        if name == "corrupt.mp4":
            target.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp4f" + b"\x00" * 64)
            print(f"wrote {target} (deliberately truncated)")
            continue
        await run_ffmpeg([*args, str(target)], timeout_s=600.0, operation=f"fixture-{name}")
        print(f"wrote {target}")


def main() -> int:
    """CLI entry.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/fixtures/media")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    asyncio.run(generate(Path(args.out), args.only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
