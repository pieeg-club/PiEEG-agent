"""Tests for the pattern persistence store."""

from pieeg_agent.decode.store import PatternStore, slugify


def test_slugify():
    assert slugify("Left Hand Imagery!") == "left-hand-imagery"
    assert slugify("  ") == "pattern"


def test_save_load_roundtrip(tmp_path):
    store = PatternStore(tmp_path)
    store.save("Calm vs Focus", {"labels": ["calm", "focus"], "score": 0.82})
    loaded = store.load("Calm vs Focus")
    assert loaded["name"] == "Calm vs Focus"
    assert loaded["slug"] == "calm-vs-focus"
    assert loaded["score"] == 0.82
    assert "saved_at" in loaded


def test_list_and_meta(tmp_path):
    store = PatternStore(tmp_path)
    store.save("alpha", {"score": 0.9, "metric": "balanced_accuracy", "n_reps": 6})
    store.save("beta", {"score": 0.7})
    assert store.list() == ["alpha", "beta"]
    meta = {m["slug"]: m for m in store.list_meta()}
    assert meta["alpha"]["score"] == 0.9
    assert meta["alpha"]["metric"] == "balanced_accuracy"


def test_exists_and_delete(tmp_path):
    store = PatternStore(tmp_path)
    assert not store.exists("ghost")
    store.save("ghost", {})
    assert store.exists("ghost")
    assert store.delete("ghost") is True
    assert store.delete("ghost") is False


def test_load_missing_returns_none(tmp_path):
    assert PatternStore(tmp_path).load("nope") is None
