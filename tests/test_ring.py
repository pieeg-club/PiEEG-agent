"""Unit tests for the ring buffer — focuses on the wrap-around logic that a
short live run never exercises (the ring only reaches ~10% fill in 8 s).
"""

from __future__ import annotations

import numpy as np

from pieeg_agent.ingest.ring import RingBuffer


def _make(n: int, ch: int, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Samples where value[s, c] = (start+s)*10 + c; timestamps = start+s."""
    idx = np.arange(start, start + n)
    data = (idx[:, None] * 10 + np.arange(ch)[None, :]).astype(np.float32)
    ts = idx.astype(np.float64)
    return data, ts


def test_basic_order_within_capacity():
    rb = RingBuffer(capacity=10, num_channels=3)
    data, ts = _make(4, 3)
    rb.push_chunk(data, ts)
    assert rb.fill == 4
    assert rb.total == 4
    out, out_ts = rb.latest(4)
    assert np.array_equal(out, data)
    assert np.array_equal(out_ts, ts)


def test_latest_clamps_to_fill():
    rb = RingBuffer(capacity=10, num_channels=2)
    rb.push_chunk(*_make(3, 2))
    out, out_ts = rb.latest(10)
    assert out.shape == (3, 2)
    assert out_ts.shape == (3,)


def test_wraparound_keeps_newest_in_order():
    # capacity 5, write 7 samples -> retains samples 2..6, oldest-first.
    rb = RingBuffer(capacity=5, num_channels=2)
    rb.push_chunk(*_make(7, 2))
    assert rb.fill == 5
    assert rb.total == 7
    out, out_ts = rb.latest(5)
    expected, expected_ts = _make(5, 2, start=2)
    assert np.array_equal(out, expected)
    assert np.array_equal(out_ts, expected_ts)
    # Most-recent single sample is index 6.
    last, last_ts = rb.latest(1)
    assert last[0, 0] == 60.0 and last[0, 1] == 61.0
    assert last_ts[0] == 6.0


def test_multiple_chunks_across_boundary():
    rb = RingBuffer(capacity=5, num_channels=1)
    rb.push_chunk(*_make(3, 1, start=0))   # 0,1,2
    rb.push_chunk(*_make(4, 1, start=3))   # 3,4,5,6 -> retains 2,3,4,5,6
    out, _ = rb.latest(5)
    assert [v[0] for v in out] == [20.0, 30.0, 40.0, 50.0, 60.0]


def test_chunk_larger_than_capacity_keeps_tail():
    rb = RingBuffer(capacity=4, num_channels=1)
    rb.push_chunk(*_make(10, 1))           # only last 4 retained: 6,7,8,9
    assert rb.fill == 4
    assert rb.total == 10
    out, out_ts = rb.latest(4)
    assert [v[0] for v in out] == [60.0, 70.0, 80.0, 90.0]
    assert list(out_ts) == [6.0, 7.0, 8.0, 9.0]


def test_empty_and_clear():
    rb = RingBuffer(capacity=4, num_channels=2)
    out, out_ts = rb.latest(3)
    assert out.shape == (0, 2) and out_ts.shape == (0,)
    rb.push_chunk(*_make(4, 2))
    assert rb.fill == 4
    rb.clear()
    assert rb.fill == 0 and rb.total == 0
    out, _ = rb.latest(1)
    assert out.shape == (0, 2)


def test_read_returns_copy_not_view():
    rb = RingBuffer(capacity=4, num_channels=1)
    rb.push_chunk(*_make(2, 1))
    out, _ = rb.latest(2)
    out[0, 0] = -999.0
    out2, _ = rb.latest(2)
    assert out2[0, 0] == 0.0  # original buffer untouched
