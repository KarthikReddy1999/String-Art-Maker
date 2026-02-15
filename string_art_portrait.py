#!/usr/bin/env python3
"""
Generate a string-art style portrait from an input image.

Example:
  python3 string_art_maker/string_art_portrait.py \
    --input couple.jpg \
    --output couple_string_art.png \
    --pins 240 \
    --lines 3200
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image #used for image processing, such as loading, resizing, and saving images.


PixelLine = Tuple[np.ndarray, np.ndarray] #used to represent the pixel coordinates of a line between two pins. 
#It consists of two numpy arrays: one for the y-coordinates and one for the x-coordinates of the pixels that form the line.


@dataclass
class Config:
    input_path: str
    output_path: str
    size: int
    pins: int
    lines: int
    min_pin_gap: int
    line_strength: float
    invert: bool
    gamma: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Create string-art portrait effect")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="string_art_output.png", help="Output PNG path")
    parser.add_argument("--size", type=int, default=720, help="Working square size in pixels")
    parser.add_argument("--pins", type=int, default=220, help="Number of virtual nails on circle")
    parser.add_argument("--lines", type=int, default=3000, help="Number of thread lines to draw")
    parser.add_argument(
        "--min-pin-gap",
        type=int,
        default=15,
        help="Minimum pin distance along circle to avoid tiny lines",
    )
    parser.add_argument(
        "--line-strength",
        type=float,
        default=0.16,
        help="How much each line darkens the image (0.01 to 1.0)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert tonal mapping when source is already dark background",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Tone curve on source darkness. >1 emphasizes shadows, <1 emphasizes mids",
    )
    args = parser.parse_args()

    if args.pins < 20:
        raise ValueError("--pins should be >= 20")
    if args.lines < 1:
        raise ValueError("--lines should be >= 1")
    if not (0.001 <= args.line_strength <= 1.0):
        raise ValueError("--line-strength must be between 0.001 and 1.0")
    if args.min_pin_gap < 1:
        raise ValueError("--min-pin-gap should be >= 1")
    if args.gamma <= 0:
        raise ValueError("--gamma should be > 0")

    return Config(
        input_path=args.input,
        output_path=args.output,
        size=args.size,
        pins=args.pins,
        lines=args.lines,
        min_pin_gap=args.min_pin_gap,
        line_strength=args.line_strength,
        invert=args.invert,
        gamma=args.gamma,
    )


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_target(cfg: Config) -> np.ndarray:
    img = Image.open(cfg.input_path).convert("L")
    img = center_crop_square(img)
    img = img.resize((cfg.size, cfg.size), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    darkness = arr if cfg.invert else (1.0 - arr)
    darkness = np.power(darkness, cfg.gamma)

    # Circular mask: keep only area reachable by string between circular pins.
    yy, xx = np.indices((cfg.size, cfg.size))
    c = (cfg.size - 1) / 2.0
    r = c - 2.0
    mask = (xx - c) ** 2 + (yy - c) ** 2 <= r * r
    darkness *= mask.astype(np.float32)
    return darkness


def make_pins(size: int, n_pins: int) -> List[Tuple[int, int]]:
    c = (size - 1) / 2.0
    r = c - 2.0
    pins: List[Tuple[int, int]] = []
    for i in range(n_pins):
        theta = 2.0 * math.pi * i / n_pins
        x = int(round(c + r * math.cos(theta)))
        y = int(round(c + r * math.sin(theta)))
        pins.append((x, y))
    return pins


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> PixelLine:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    xs: List[int] = []
    ys: List[int] = []

    while True:
        xs.append(x0)
        ys.append(y0)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)


def precompute_lines(pins: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], PixelLine]:
    line_map: Dict[Tuple[int, int], PixelLine] = {}
    for i in range(len(pins)):
        x0, y0 = pins[i]
        for j in range(i + 1, len(pins)):
            x1, y1 = pins[j]
            line_map[(i, j)] = bresenham_line(x0, y0, x1, y1)
    return line_map


def pin_distance_circular(a: int, b: int, n: int) -> int:
    d = abs(a - b)
    return min(d, n - d)


def get_line_pixels(lines: Dict[Tuple[int, int], PixelLine], a: int, b: int) -> PixelLine:
    if a < b:
        return lines[(a, b)]
    return lines[(b, a)]


def generate(cfg: Config) -> Tuple[np.ndarray, List[int]]:
    target = load_target(cfg)
    pins = make_pins(cfg.size, cfg.pins)
    lines = precompute_lines(pins)

    residual = target.copy()
    render = np.ones_like(target, dtype=np.float32)
    order = [0]
    current = 0

    candidates_by_pin: List[List[int]] = []
    for i in range(cfg.pins):
        candidates: List[int] = []
        for j in range(cfg.pins):
            if j == i:
                continue
            if pin_distance_circular(i, j, cfg.pins) < cfg.min_pin_gap:
                continue
            candidates.append(j)
        candidates_by_pin.append(candidates)

    for step in range(cfg.lines):
        best_pin = -1
        best_score = -1e18

        for cand in candidates_by_pin[current]:
            ys, xs = get_line_pixels(lines, current, cand)
            score = float(residual[ys, xs].sum())
            if score > best_score:
                best_score = score
                best_pin = cand

        if best_pin == -1:
            break

        ys, xs = get_line_pixels(lines, current, best_pin)
        residual[ys, xs] -= cfg.line_strength
        render[ys, xs] = np.maximum(0.0, render[ys, xs] - cfg.line_strength)

        order.append(best_pin)
        current = best_pin

        if (step + 1) % 200 == 0:
            print(f"Drawn {step + 1}/{cfg.lines} lines...")

    return render, order


def save_output(render: np.ndarray, output_path: str) -> None:
    out = np.clip(render * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, mode="L")
    img.save(output_path)


def main() -> None:
    cfg = parse_args()
    print("Preparing string-art... this can take a little while for large settings.")
    render, order = generate(cfg)
    save_output(render, cfg.output_path)
    print(f"Saved: {cfg.output_path}")
    print(f"Pin path length: {len(order)}")


if __name__ == "__main__":
    main()
