#!/usr/bin/env python3
"""Extract reference-software cursor positions from replay screenshots.

The green crosshair is a reference-controller output pseudo-label. It is NOT
ground truth for the invisible game target. Keep that distinction in reports.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


SEQ_RE = re.compile(r"seq(\d+)", re.IGNORECASE)
TIME_RE = re.compile(r"_(\d{6})_(\d{3})_seq", re.IGNORECASE)


def parse_stamp(path: Path) -> tuple[int | None, int | None]:
    seq_match = SEQ_RE.search(path.name)
    time_match = TIME_RE.search(path.name)
    seq = int(seq_match.group(1)) if seq_match else None
    if not time_match:
        return seq, None
    hms, millis = time_match.groups()
    stamp_ms = (
        int(hms[0:2]) * 3_600_000
        + int(hms[2:4]) * 60_000
        + int(hms[4:6]) * 1_000
        + int(millis)
    )
    return seq, stamp_ms


def find_candidates(frame: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([35, 100, 90], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    # Ignore title bar, footer, and outer window frame.
    mask[:25, :] = 0
    mask[530:, :] = 0
    mask[:, :8] = 0
    mask[:, 750:] = 0

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        # The recorded crosshair is approximately 34x34 and 430 green pixels.
        # This excludes the smaller green menu icon seen before a game round.
        if not (29 <= w <= 38 and 29 <= h <= 38):
            continue
        if not (390 <= area <= 560) or abs(w - h) > 5:
            continue

        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        # A crosshair has both a circular ring and horizontal/vertical strokes.
        # Require green pixels in the central horizontal and vertical bands.
        cy, cx = h // 2, w // 2
        horizontal = int(component[max(0, cy - 2):cy + 3, :].sum())
        vertical = int(component[:, max(0, cx - 2):cx + 3].sum())
        if horizontal < 12 or vertical < 12:
            continue

        shape_penalty = abs(w - 34) + abs(h - 34) + abs(area - 432) / 35.0
        candidates.append({
            "x": float(centers[label][0]),
            "y": float(centers[label][1]),
            "area": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "shape_penalty": float(shape_penalty),
        })
    return candidates


def choose_candidate(candidates: list[dict], previous: dict | None) -> dict | None:
    if not candidates:
        return None
    if previous is None or len(candidates) == 1:
        return min(candidates, key=lambda c: c["shape_penalty"])

    # Prefer continuity when more than one green component is present. A large
    # jump is allowed when it is the only plausible crosshair (round start).
    near = [
        c for c in candidates
        if np.hypot(c["x"] - previous["x"], c["y"] - previous["y"]) <= 120
    ]
    if near:
        return min(
            near,
            key=lambda c: c["shape_penalty"]
            + 0.06 * np.hypot(c["x"] - previous["x"], c["y"] - previous["y"]),
        )
    return min(candidates, key=lambda c: c["shape_penalty"])


def extract(root: Path, output: Path) -> int:
    images = sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: (str(p.parent), parse_stamp(p)[0] or -1),
    )
    rows = []
    previous_by_session: dict[str, dict] = {}
    for path in images:
        seq, stamp_ms = parse_stamp(path)
        if seq is None:
            continue
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (563, 759):
            continue
        session = path.parent.name
        candidates = find_candidates(frame)
        chosen = choose_candidate(candidates, previous_by_session.get(session))
        if chosen is not None:
            previous_by_session[session] = chosen
        rows.append({
            "session": session,
            "seq": seq,
            "timestamp_ms": stamp_ms,
            "x": "" if chosen is None else f"{chosen['x']:.3f}",
            "y": "" if chosen is None else f"{chosen['y']:.3f}",
            "area": "" if chosen is None else chosen["area"],
            "bbox_x": "" if chosen is None else chosen["bbox_x"],
            "bbox_y": "" if chosen is None else chosen["bbox_y"],
            "bbox_w": "" if chosen is None else chosen["bbox_w"],
            "bbox_h": "" if chosen is None else chosen["bbox_h"],
            "candidate_count": len(candidates),
            "label_semantics": "reference_cursor_pseudo_label",
            "source_file": str(path.relative_to(root)),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "session", "seq", "timestamp_ms", "x", "y", "area",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "candidate_count",
        "label_semantics", "source_file",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(row["x"] != "" for row in rows)
    print(f"frames={len(rows)} cursor_labels={valid} output={output}")
    print("WARNING: cursor labels reproduce the reference software's mouse output;")
    print("         they are not independent ground truth for the invisible target.")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reference_cursor_labels.csv"))
    args = parser.parse_args()
    if not args.frames_dir.is_dir():
        raise SystemExit(f"frames directory not found: {args.frames_dir}")
    extract(args.frames_dir, args.output)


if __name__ == "__main__":
    main()
