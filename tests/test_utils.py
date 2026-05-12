"""Tests for utils.py — written BEFORE implementation (TDD Red phase)."""
import json
import os
import time
import pytest
from src.utils import calculate_token_usage, log_metrics, Timer


# --- calculate_token_usage ---

def test_token_usage_returns_int():
    assert isinstance(calculate_token_usage("hello world"), int)


def test_token_usage_empty_string():
    assert calculate_token_usage("") == 0


def test_token_usage_scales_with_length():
    short = calculate_token_usage("hello")
    long = calculate_token_usage("hello " * 100)
    assert long > short


def test_token_usage_non_negative():
    assert calculate_token_usage("any text here") >= 0
    assert calculate_token_usage("") >= 0


def test_token_usage_reasonable_estimate():
    """~4 chars/token: 400-char string should estimate roughly 100 tokens."""
    text = "a" * 400
    tokens = calculate_token_usage(text)
    assert 80 <= tokens <= 120


# --- log_metrics ---

def test_log_metrics_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = {"latency": 1.234, "chunks": 5, "questions": 3}
    log_metrics(metrics)
    log_files = list((tmp_path / "logs").glob("metrics_*.json"))
    assert len(log_files) == 1


def test_log_metrics_file_valid_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = {"latency": 0.5, "cache_hit": True}
    log_metrics(metrics)
    log_file = next((tmp_path / "logs").glob("metrics_*.json"))
    with open(log_file) as f:
        data = json.load(f)
    assert data == metrics


def test_log_metrics_empty_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_metrics({})  # should not raise


def test_log_metrics_float_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = {"p50": 0.123456, "p99": 1.987654}
    log_metrics(metrics)  # should not raise


# --- Timer ---

def test_timer_measures_elapsed():
    with Timer("test") as t:
        time.sleep(0.05)
    assert t.interval >= 0.04


def test_timer_interval_attribute_set():
    with Timer() as t:
        pass
    assert hasattr(t, "interval")
    assert t.interval >= 0


def test_timer_with_name(capsys):
    with Timer(name="my_block"):
        pass
    captured = capsys.readouterr()
    assert "my_block" in captured.out


def test_timer_no_name_no_output(capsys):
    with Timer():
        pass
    captured = capsys.readouterr()
    assert captured.out == ""


def test_timer_nested():
    with Timer("outer") as outer:
        with Timer("inner") as inner:
            time.sleep(0.02)
    assert outer.interval >= inner.interval
