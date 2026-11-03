#!/usr/bin/env python3
"""Offline replay evaluator for tracker output versus reference cursor pseudo-labels.

The green cursor is the reference assistant's output, not independent ground
truth for the invisible game target. Metrics must be reported with that caveat.
This script never sends mouse or keyboard input.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from extract_reference_cursor_labels import extract, parse_stamp
from tracker import WhiteTargetTracker


def load_labels(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("x") or not row.get("y"):
                continue
            labels[(row["session"], int(row["seq"]))] = (
                float(row["x"]), float(row["y"])
            )
    return labels


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["reference_x"] != "" and r["prediction_x"] != ""]
    errors = [float(r["pseudo_label_error_px"]) for r in valid]
    predicted = [
        float(r["pseudo_label_error_px"]) for r in valid
        if r["mode"] in {"flow", "kalman_prediction"}
    ]
    visible = [r for r in rows if r["mode"] == "visible"]
    return {
        "frames": len(rows),
        "visible_frames": len(visible),
        "flow_frames": sum(r["mode"] == "flow" for r in rows),
        "kalman_prediction_frames": sum(r["mode"] == "kalman_prediction" for r in rows),
        "missing_frames": sum(r["mode"] == "missing" for r in rows),
        "visible_safe_point_inside_ratio": (
            sum(r["safe_point_inside"] == "1" for r in visible) / len(visible)
            if visible else None
        ),
        "reference_cursor_pseudo_label_matches": len(valid),
        "pseudo_label_error_p50_px": percentile(errors, 50),
        "pseudo_label_error_p90_px": percentile(errors, 90),
        "pseudo_label_error_p95_px": percentile(errors, 95),
        "prediction_only_pseudo_label_error_p50_px": percentile(predicted, 50),
        "prediction_only_pseudo_label_error_p90_px": percentile(predicted, 90),
        "prediction_only_pseudo_label_error_p95_px": percentile(predicted, 95),
        "warning": (
            "Errors compare this tracker to the reference assistant cursor output. "
            "They are not ground-truth errors for the hidden target."
        ),
    }


def run_replay(
    frames_dir: Path,
    labels: dict[tuple[str, int], tuple[float, float]],
    threshold: int,
    min_area: int,
    max_missing: int,
) -> list[dict]:
    paths = sorted(
        (
            p for p in frames_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and re.search(r"seq\d+", p.name, re.IGNORECASE)
        ),
        key=lambda p: (str(p.parent), parse_stamp(p)[0] or -1),
    )
    by_session: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_session[path.parent.name].append(path)

    rows = []
    for session, session_paths in sorted(by_session.items()):
        tracker = WhiteTargetTracker(max_missing=max_missing, use_gpu=False)
        previous_timestamp = None
        for path in session_paths:
            seq, timestamp_ms = parse_stamp(path)
            if seq is None:
                continue
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if previous_timestamp is None or timestamp_ms is None:
                dt = 1.0
            else:
                dt = float(np.clip(
                    ((timestamp_ms - previous_timestamp) / 1000.0) * 30.0,
                    0.20, 4.0,
                ))
            previous_timestamp = timestamp_ms

            point = tracker.update(rgb, threshold, min_area, dt=dt)
            reference = labels.get((session, seq))
            if tracker.last_detected:
                mode = "visible"
            elif point is not None and tracker.flow_confidence >= 0.20:
                mode = "flow"
            elif point is not None:
                mode = "kalman_prediction"
            else:
                mode = "missing"

            inside = False
            if point is not None and tracker.last_detected and tracker.last_mask is not None:
                px, py = point
                if 0 <= py < tracker.last_mask.shape[0] and 0 <= px < tracker.last_mask.shape[1]:
                    inside = bool(tracker.last_mask[py, px] > 0)

            error = ""
            if point is not None and reference is not None:
                error = float(np.hypot(
                    point[0] - reference[0], point[1] - reference[1]
                ))
            rows.append({
                "session": session,
                "seq": seq,
                "timestamp_ms": "" if timestamp_ms is None else timestamp_ms,
                "mode": mode,
                "prediction_x": "" if point is None else point[0],
                "prediction_y": "" if point is None else point[1],
                "reference_x": "" if reference is None else reference[0],
                "reference_y": "" if reference is None else reference[1],
                "pseudo_label_error_px": error,
                "tracker_confidence": round(float(tracker.last_confidence), 4),
                "flow_confidence": round(float(tracker.flow_confidence), 4),
                "missing_frames": tracker.missing,
                "safe_point_inside": "1" if inside else "0",
                "source_file": str(path.relative_to(frames_dir)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("tracker_replay.csv"))
    parser.add_argument("--threshold", type=int, default=215)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--max-missing", type=int, default=180)
    args = parser.parse_args()

    if not args.frames_dir.is_dir():
        raise SystemExit(f"frames directory not found: {args.frames_dir}")

    labels_path = args.labels_csv or args.output_csv.with_name("reference_cursor_labels.csv")
    if args.labels_csv is None:
        extract(args.frames_dir, labels_path)
    labels = load_labels(labels_path)

    rows = run_replay(
        args.frames_dir, labels, args.threshold, args.min_area, args.max_missing
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)

    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row["session"]].append(row)
    report = {
        "overall": summarize(rows),
        "sessions": {name: summarize(items) for name, items in sorted(by_session.items())},
        "output_csv": str(args.output_csv),
        "labels_csv": str(labels_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
