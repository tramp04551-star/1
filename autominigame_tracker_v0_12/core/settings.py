"""Small JSON settings store for the desktop UI."""
from __future__ import annotations
import json
import os
from pathlib import Path

DEFAULT_SETTINGS = {
    "window_title": "Artale",
    "fps": 30.0,
    "follow_mouse": True,
    "enable_events": True,
    "enable_ocr": False,
    "max_lost_frames": 8,
    "acquire_frames": 4,
    "cursor_smoothing": 0.0,
    "cursor_deadband_px": 1.5,
    "cursor_max_speed_px_s": 0.0,
}


def settings_path() -> Path:
    if os.name == "nt":
        root = Path(os.getenv("APPDATA", Path.home()))
    else:
        root = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "AutoMinigameTracker" / "settings.json"


def load_settings() -> dict:
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(DEFAULT_SETTINGS)
        merged.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
        return merged
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(data: dict) -> Path:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(DEFAULT_SETTINGS)
    merged.update({k: data.get(k, v) for k, v in DEFAULT_SETTINGS.items()})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path