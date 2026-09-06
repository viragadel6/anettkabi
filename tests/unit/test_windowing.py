"""Window planning and equal-power crossfade math (torch-dependent)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _windowing():
    pytest.importorskip("torch")
    from app.ml.windowing import equal_power_crossfade_add, plan_windows, target_samples

    return equal_power_crossfade_add, plan_windows, target_samples


def test_single_window_when_short() -> None:
    _fade, plan_windows, _samples = _windowing()
    plans = plan_windows(8.0, 10.0, 1.0)
    assert len(plans) == 1
    assert plans[0].start_s == 0.0
    assert plans[0].length_s == 8.0
    assert plans[0].crop_start_s == 0.0
    assert plans[0].crop_end_s == 8.0


def test_multi_window_schedule() -> None:
    _fade, plan_windows, _samples = _windowing()
    plans = plan_windows(25.0, 10.0, 1.0)
    assert len(plans) == 3
    assert [round(plan.start_s, 6) for plan in plans] == [0.0, 9.0, 18.0]
    assert plans[-1].length_s == pytest.approx(7.0)


def test_crops_trim_half_overlap_on_interior_boundaries() -> None:
    _fade, plan_windows, _samples = _windowing()
    plans = plan_windows(25.0, 10.0, 1.0)
    assert plans[0].crop_end_s == pytest.approx(9.5)
    assert plans[1].crop_start_s == pytest.approx(9.5)
    assert plans[1].crop_end_s == pytest.approx(18.5)
    assert plans[2].crop_start_s == pytest.approx(18.5)


def test_overlap_guard() -> None:
    _fade, plan_windows, _samples = _windowing()
    with pytest.raises(ValueError, match="overlap"):
        plan_windows(10.0, 10.0, 5.0)
    with pytest.raises(ValueError, match="duration"):
        plan_windows(0.0, 10.0, 1.0)


def test_equal_power_crossfade_preserves_constant_tone() -> None:
    crossfade, _plan, _samples = _windowing()
    sample_rate = 16000
    fade = 1600
    total = torch.zeros(1, sample_rate * 2)
    first = torch.ones(1, sample_rate * 2)
    total = crossfade(total, first, 0, fade)
    second = torch.ones(1, sample_rate * 2)
    total = crossfade(total, second, sample_rate, fade)
    overlap = total[0, sample_rate : sample_rate + fade]
    assert torch.allclose(overlap, torch.ones_like(overlap), atol=0.02)


def test_crossfade_validates_shapes() -> None:
    crossfade, _plan, _samples = _windowing()
    with pytest.raises(ValueError, match="channel"):
        crossfade(torch.zeros(2, 10), torch.zeros(1, 10), 0, 4)
    with pytest.raises(ValueError, match="bounds"):
        crossfade(torch.zeros(1, 10), torch.zeros(1, 20), 0, 4)


def test_target_samples_rounding() -> None:
    _fade, _plan, target_samples = _windowing()
    assert target_samples(1.0, 16000) == 16000
    assert target_samples(10.5, 16000) == 168000
    assert target_samples(1.0 / 3, 48000) == round(1.0 / 3 * 48000)


def test_windows_cover_full_timeline() -> None:
    _fade, plan_windows, _samples = _windowing()
    plans = plan_windows(63.7, 10.0, 1.0)
    assert plans[0].start_s == 0.0
    assert plans[-1].start_s + plans[-1].length_s == pytest.approx(63.7)
