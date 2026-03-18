#!/usr/bin/env python3
"""Generate a small RGB thumbnail (bands 5-3-1 by default) from a multispectral GeoTIFF.

Designed for quick previews in cnn_mappingAI_viewer.html.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PNG thumbnail from a multispectral GeoTIFF")
    parser.add_argument("--input", required=True, help="Input GeoTIFF (ideally EPSG:4326 COG)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--bands",
        default="5,3,1",
        help="Comma-separated 1-based band numbers to map to R,G,B (default: 5,3,1)",
    )
    parser.add_argument("--size", type=int, default=256, help="Target max width/height in pixels (default: 256)")
    parser.add_argument(
        "--pmin",
        type=float,
        default=2.0,
        help="Lower percentile for contrast stretch (default: 2)",
    )
    parser.add_argument(
        "--pmax",
        type=float,
        default=98.0,
        help="Upper percentile for contrast stretch (default: 98)",
    )
    return parser.parse_args()


def _safe_percentile(values: np.ndarray, p: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, p))


def _stretch_to_byte(band: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    lo = _safe_percentile(band, pmin)
    hi = _safe_percentile(band, pmax)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)

    scaled = (band - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    args = parse_args()

    if iio is None:
        raise RuntimeError(
            "imageio is required for PNG output but could not be imported. "
            "Install imageio or run this script inside the project conda env."
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    band_numbers = [int(x.strip()) for x in str(args.bands).split(",") if x.strip()]
    if len(band_numbers) != 3:
        raise ValueError("--bands must specify exactly 3 band numbers, e.g. '5,3,1'.")

    with rasterio.open(input_path) as src:
        # Preserve aspect ratio; keep the larger dimension equal to args.size.
        scale = args.size / max(src.width, src.height)
        out_w = max(1, int(math.ceil(src.width * scale)))
        out_h = max(1, int(math.ceil(src.height * scale)))

        rgb = []
        for b in band_numbers:
            arr = src.read(
                b,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)

            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)

            rgb.append(_stretch_to_byte(arr, args.pmin, args.pmax))

        rgb_img = np.stack(rgb, axis=-1)

    iio.imwrite(output_path, rgb_img)
    print(f"Wrote thumbnail: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
