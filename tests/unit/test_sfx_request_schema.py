"""SfxRequest schema validation ranges."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas.sfx_request import SfxRequest


def test_minimal_request_uses_defaults() -> None:
    request = SfxRequest(video="https://cdn.example.com/a.mp4", prompt="boom")
    assert request.num_inference_steps == 25
    assert request.guidance_scale == 4.5
    assert request.video_handling in ("copy", "reencode")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_inference_steps", 0),
        ("num_inference_steps", 101),
        ("guidance_scale", -0.1),
        ("guidance_scale", 15.5),
        ("duration", 0),
        ("duration", -5),
        ("start_time", -1),
        ("sfx_gain_db", 13),
        ("duck_ratio", 0.5),
        ("duck_attack_ms", 0),
        ("duck_release_ms", 5),
        ("target_loudness_lufs", -41),
        ("true_peak_db", 1),
    ],
)
def test_out_of_range_values_rejected(field: str, value: float) -> None:
    payload = {"video": "https://cdn.example.com/a.mp4", "prompt": "x", field: value}
    with pytest.raises(ValidationError):
        SfxRequest(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_inference_steps", 1),
        ("num_inference_steps", 100),
        ("guidance_scale", 0.0),
        ("duration", 0.5),
        ("start_time", 120),
        ("sfx_gain_db", -30),
        ("duck_ratio", 20.0),
        ("duck_release_ms", 2000),
    ],
)
def test_boundary_values_accepted(field: str, value: float) -> None:
    payload = {"video": "https://cdn.example.com/a.mp4", "prompt": "x", field: value}
    request = SfxRequest(**payload)
    assert getattr(request, field) == value


def test_audio_mode_enum_enforced() -> None:
    with pytest.raises(ValidationError):
        SfxRequest(video="https://cdn.example.com/a.mp4", prompt="x", audio_mode="blend")


def test_prompt_and_video_default_empty() -> None:
    request = SfxRequest()
    assert request.video is None
    assert request.prompt == ""
    assert request.seed == -1


def test_metadata_object_default() -> None:
    request = SfxRequest(video="https://cdn.example.com/a.mp4", prompt="x")
    assert request.metadata == {}
    request = SfxRequest(video="https://cdn.example.com/a.mp4", prompt="x", metadata={"k": 1})
    assert request.metadata == {"k": 1}
