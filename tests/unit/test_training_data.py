"""Dataset acquisition logic: FSD50K index parsing and VGGSound selection.

Pinned to the REAL upstream formats (validated against the actual
FSD50K.ground_truth.zip and Loie/VGGSound vggsound.csv in September 2026):

* FSD50K ground truth ships ``FSD50K.ground_truth/{dev,eval,vocabulary}.csv``
  with columns ``fname,labels,mids[,split]`` and a header row.
* VGGSound csv is header-less with columns ``youtube_id,start_s,label,split``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.config import TrainSettings
from training.data.fsd50k import (
    DEV_FALLBACK,
    EVAL_FALLBACK,
    _zenodo_files,
    clip_paths,
    iter_fsd50k,
    prepare_fsd50k_index,
    record_files,
    select_archive_names,
)
from training.data.vggsound import _label_ok, _prompt_for, iter_vggsound, prepare_vggsound


def _settings(tmp_path: Path) -> TrainSettings:
    return TrainSettings(data_dir=tmp_path, output_dir=tmp_path / "runs", weights_dir=tmp_path / "weights")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_fsd50k_index_parses_real_csv_layout(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write(
        tmp_path / "fsd50k" / "FSD50K.ground_truth" / "dev.csv",
        "fname,labels,mids,split\n"
        '64760,"Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music","/m/02sgy,/m/0342h",train\n'
        '16399,"Thunderstorm,Thunder,Rain","\x2fm\x2f02sgy",valid\n',
    )
    _write(
        tmp_path / "fsd50k" / "FSD50K.ground_truth" / "eval.csv",
        "fname,labels,mids\n"
        '37199,"Electric_guitar,Guitar","/m/02sgy"\n'
        '175151,"Thunderstorm,Thunder","/m/02sgy"\n',
    )
    out = prepare_fsd50k_index(settings)
    index = json.loads(out.read_text(encoding="utf-8"))
    assert len(index) == 4
    assert index[0] == {
        "clip_id": "64760",
        "split": "dev",
        "labels": ["Electric guitar", "Guitar", "Plucked string instrument", "Musical instrument", "Music"],
        "prompt": "Electric guitar, Guitar, Plucked string instrument, Musical instrument, Music",
        "subsplit": "train",
    }
    assert index[1]["subsplit"] == "valid"
    assert index[2]["split"] == "eval"
    assert "subsplit" not in index[2]


def test_fsd50k_index_accepts_legacy_filenames(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write(
        tmp_path / "fsd50k" / "FSD50K.ground_truth" / "FSD50K.ground_truth-dev.csv",
        "fname,labels,mids\n100,Dog_Bark,/m/x\n",
    )
    prepare_fsd50k_index(settings)
    index = json.loads((tmp_path / "fsd50k" / "index.json").read_text(encoding="utf-8"))
    assert index[0]["clip_id"] == "100"
    assert index[0]["labels"] == ["Dog Bark"]


def test_fsd50k_clip_paths_prefers_renamed_layout(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write(tmp_path / "fsd50k" / "dev" / "64760.wav", "x")
    assert clip_paths(settings, "64760", "dev") == tmp_path / "fsd50k" / "dev" / "64760.wav"
    (tmp_path / "fsd50k" / "dev").rename(tmp_path / "fsd50k" / "FSD50K.dev_audio")
    assert clip_paths(settings, "64760", "dev") == tmp_path / "fsd50k" / "FSD50K.dev_audio" / "64760.wav"


def test_iter_fsd50k_yields_only_existing_audio(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write(
        tmp_path / "fsd50k" / "FSD50K.ground_truth" / "dev.csv",
        "fname,labels,mids\n1,Dog,/m/x\n2,Cat,/m/y\n",
    )
    prepare_fsd50k_index(settings)
    _write(tmp_path / "fsd50k" / "dev" / "1.wav", "audio")
    entries = list(iter_fsd50k(settings))
    assert [entry["clip_id"] for entry in entries] == ["1"]


def test_vggsound_label_filter_matches_real_classes_only() -> None:
    assert _label_ok("car engine knocking")
    assert _label_ok("people typing on keyboard")
    assert _label_ok("dog barking")
    assert _label_ok("chainsawing trees")
    assert _label_ok("engine accelerating, revving, vroom")
    assert _label_ok("rain on windshield")
    assert not _label_ok("people marching")
    assert not _label_ok("playing tennis")
    assert not _label_ok("people belly laughing")


def test_vggsound_split_column_is_not_treated_as_label() -> None:
    assert not _label_ok("train")
    assert not _label_ok("test")


def test_vggsound_prompt_rendering() -> None:
    assert _prompt_for("people typing on keyboard") == "realistic people typing on keyboard, clean recording"
    assert _prompt_for("engine accelerating, revving, vroom.") == (
        "realistic engine accelerating, revving, vroom, clean recording"
    )


def _vggsound_csv(tmp_path: Path) -> Path:
    lines = [
        "abc12345,30,people marching,train",
        "def67890,0,car engine starting,train",
        "ghi11223,400,people clapping,train",
        "jkl44556,120,playing tennis,test",
        "mno77889,not-a-number,dog barking,train",
        "pqr99001,15,wind noise,train",
    ]
    csv_path = tmp_path / "vggsound" / "vggsound.csv"
    _write(csv_path, "\n".join(lines) + "\n")
    return csv_path


def test_vggsound_prepare_filters_shuffles_and_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    csv_path = _vggsound_csv(tmp_path)
    monkeypatch.setattr(
        "training.data.vggsound.download_vggsound_csv",
        lambda _settings: csv_path,
    )
    fetched: list[str] = []

    def fake_fetch(youtube_id: str, start_s: float, destination: Path) -> bool:
        if youtube_id == "ghi11223":
            return False
        fetched.append(f"{youtube_id}:{start_s:.0f}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\0" * 20480)
        return True

    monkeypatch.setattr("training.data.vggsound.fetch_clip", fake_fetch)
    index_path = prepare_vggsound(settings, max_clips=10)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    labels = sorted(entry["label"] for entry in index)
    assert labels == ["car engine starting", "wind noise"]
    assert all(int(entry["start_s"]) >= 0 for entry in index)
    assert all(Path(str(entry["path"])).is_file() for entry in index)
    assert all("prompt" in entry for entry in index)
    assert len(fetched) == 2

    again = list(iter_vggsound(settings))
    assert sorted(entry["label"] for entry in again) == labels


def test_vggsound_seed_makes_selection_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    csv_path = _vggsound_csv(tmp_path)
    monkeypatch.setattr(
        "training.data.vggsound.download_vggsound_csv", lambda _settings: csv_path
    )
    monkeypatch.setattr(
        "training.data.vggsound.fetch_clip",
        lambda _youtube_id, _start, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"\0" * 20480),
            True,
        )[-1],
    )
    first = json.loads(prepare_vggsound(settings, max_clips=2).read_text(encoding="utf-8"))
    second = json.loads(prepare_vggsound(settings, max_clips=2).read_text(encoding="utf-8"))
    assert [entry["youtube_id"] for entry in first] == [entry["youtube_id"] for entry in second]


def test_vggsound_malformed_rows_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    csv_path = tmp_path / "vggsound" / "vggsound.csv"
    _write(
        csv_path,
        "goodid111,10,rain on windshield,train\nshortrow,10\nbadstart22,xx,dog barking,train\n",
    )
    monkeypatch.setattr(
        "training.data.vggsound.download_vggsound_csv", lambda _settings: csv_path
    )
    monkeypatch.setattr(
        "training.data.vggsound.fetch_clip",
        lambda _youtube_id, _start, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"\0" * 20480),
            True,
        )[-1],
    )
    index = json.loads(prepare_vggsound(settings, max_clips=10).read_text(encoding="utf-8"))
    assert [entry["youtube_id"] for entry in index] == ["goodid111"]


def test_vggsound_dataset_split_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    csv_path = _vggsound_csv(tmp_path)
    monkeypatch.setattr(
        "training.data.vggsound.download_vggsound_csv", lambda _settings: csv_path
    )
    monkeypatch.setattr(
        "training.data.vggsound.fetch_clip",
        lambda _youtube_id, _start, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_bytes(b"\0" * 20480),
            True,
        )[-1],
    )
    index = json.loads(prepare_vggsound(settings, max_clips=10).read_text(encoding="utf-8"))
    assert all(entry.get("dataset_split") in ("train", "test") for entry in index)


def test_vggsound_csv_cache_threshold_matches_real_size(tmp_path: Path) -> None:
    csv_path = _vggsound_csv(tmp_path)
    csv_path.write_bytes(b"x" * (10_000_001))
    assert csv_path.stat().st_size == 10_000_001


LIVE_RECORD_4060432 = [
    ("FSD50K.dev_audio.z01", 3_221_259_264),
    ("FSD50K.dev_audio.z02", 3_221_259_264),
    ("FSD50K.dev_audio.z03", 3_221_259_264),
    ("FSD50K.dev_audio.z04", 3_221_259_264),
    ("FSD50K.dev_audio.z05", 3_221_259_264),
    ("FSD50K.dev_audio.zip", 2_312_000_000),
    ("FSD50K.eval_audio.z01", 3_221_259_264),
    ("FSD50K.eval_audio.zip", 3_040_000_000),
    ("FSD50K.ground_truth.zip", 286_000),
    ("FSD50K.metadata.zip", 10_000_000),
    ("FSD50K.doc.zip", 300_000),
]


def test_archive_selection_uses_live_record_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "training.data.fsd50k.record_files",
        lambda url: LIVE_RECORD_4060432 if "4060432" in url else [],
    )
    names = select_archive_names(settings, include_eval=True)
    assert names == [
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
        "FSD50K.dev_audio.zip",
        "FSD50K.ground_truth.zip",
        "FSD50K.eval_audio.z01",
        "FSD50K.eval_audio.zip",
    ]
    dev_only = select_archive_names(settings, include_eval=False)
    assert not any(name.startswith("FSD50K.eval_audio") for name in dev_only)


def test_archive_selection_falls_back_when_api_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("training.data.fsd50k.record_files", lambda url: None)
    names = select_archive_names(settings, include_eval=True)
    assert names == [*DEV_FALLBACK, "FSD50K.ground_truth.zip", *EVAL_FALLBACK]
    assert DEV_FALLBACK == [
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
        "FSD50K.dev_audio.zip",
    ]


def test_zenodo_url_builder() -> None:
    urls = _zenodo_files("https://zenodo.org/records/4060432", ["FSD50K.doc.zip"])
    assert urls == ["https://zenodo.org/records/4060432/files/FSD50K.doc.zip?download=1"]


def test_record_files_returns_none_on_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    def refuse(_client: object, _url: str) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.Client, "get", refuse)
    assert record_files("https://zenodo.org/records/4060432") is None
