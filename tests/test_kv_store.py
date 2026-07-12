"""
Unit tests for the SQLite-backed kv_store module.

Uses a temporary DB path (via monkeypatch) so tests never touch the real
data/app_state.db file used by the running application.
"""

import researchmind.kv_store as kv_store


def test_set_and_get_string_value(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    kv_store.set_value("ns1", "key1", "hello")
    assert kv_store.get_value("ns1", "key1") == "hello"


def test_set_and_get_dict_value(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    kv_store.set_value("ns1", "key2", {"a": 1, "b": "two"})
    assert kv_store.get_value("ns1", "key2", as_json=True) == {"a": 1, "b": "two"}


def test_get_missing_key_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    assert kv_store.get_value("ns1", "nonexistent") is None


def test_overwrite_value(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    kv_store.set_value("ns1", "key3", "first")
    kv_store.set_value("ns1", "key3", "second")
    assert kv_store.get_value("ns1", "key3") == "second"


def test_namespaces_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    kv_store.set_value("nsA", "same_key", "valueA")
    kv_store.set_value("nsB", "same_key", "valueB")
    assert kv_store.get_value("nsA", "same_key") == "valueA"
    assert kv_store.get_value("nsB", "same_key") == "valueB"