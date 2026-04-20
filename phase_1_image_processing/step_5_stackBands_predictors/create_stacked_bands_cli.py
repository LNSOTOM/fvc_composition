#!/usr/bin/env python3
"""Create a stacked predictor raster by selecting bands from a multispectral raster.

This is the parameterized replacement for `5_2_imageProcessing_stackedBandsRaster.py`.
It supports either a single input raster or a directory of rasters, preserves
georeferencing, copies band metadata where available, and can write either a
compressed GeoTIFF or a Cloud-Optimized GeoTIFF.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window
from tqdm import tqdm


DEFAULT_BAND_SELECTION = [2, 4, 6, 8, 10]
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_OUTPUT_PREFIX = "stacked_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stacked predictor raster from one raster or a directory of rasters"
    )
    parser.add_argument("--input", required=True, help="Input .tif file or directory containing .tif rasters")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .tif path (single input) or output directory (directory input)",
    )
    parser.add_argument(
        "--bands",
        default=",".join(str(value) for value in DEFAULT_BAND_SELECTION),
        help="Comma-separated 1-based band numbers to keep (default: 2,4,6,8,10)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Processing window size in pixels (default: 1024)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Output raster block size in pixels (default: 512)",
    )
    parser.add_argument(
        "--output-driver",
        choices=["GTiff", "COG"],
        default="GTiff",
        help="Output driver (default: GTiff)",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for output files when --input is a directory (default: stacked_)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def _pick_first(tags: dict, keys: list[str]) -> str | None:
    for key in keys:
        for candidate in (key, key.lower(), key.upper()):
            value = tags.get(candidate)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return None


def _format_nm(value: str) -> str | None:
    try:
        nm = float(str(value).strip())
    except Exception:
        return None
    if nm <= 0:
        return None
    if abs(nm - round(nm)) < 1e-6:
        return str(int(round(nm)))
    return f"{nm:g}"


def _band_description_from_metadata(src: rasterio.io.DatasetReader, band_index: int) -> str:
    try:
        descriptions = list(src.descriptions or [])
        desc = descriptions[band_index - 1] if band_index - 1 < len(descriptions) else None
        if desc and str(desc).strip():
            return str(desc).strip()
    except Exception:
        pass

    try:
        tags = src.tags(band_index) or {}
        description = _pick_first(tags, ["description", "band_description", "long_name"])
        if description:
            return description
        name = _pick_first(tags, ["band_name", "bandname", "name"])
        wavelength = _pick_first(tags, ["wavelength", "center_wavelength", "central_wavelength"])
        nm = _format_nm(wavelength) if wavelength else None
        if name and nm:
            return f"Band {band_index:02d}: {name} [{nm} nm]"
        if name:
            return f"Band {band_index:02d}: {name}"
        if nm:
            return f"Band {band_index:02d}: [{nm} nm]"
    except Exception:
        pass

    return f"Band {band_index:02d}"


def _parse_band_list(raw: str) -> list[int]:
    bands = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not bands:
        raise ValueError("--bands must contain at least one band number")
    if min(bands) < 1:
        raise ValueError("Band numbers must be 1-based positive integers")
    return bands


def _normalize_block_size(value: int) -> int:
    value = max(16, min(512, int(value)))
    value = (value // 16) * 16
    return value if value >= 16 else 16


def _iter_input_rasters(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.tif") if path.is_file())
    return [input_path]


def _resolve_output_path(
    *,
    input_raster: Path,
    output: Path,
    is_batch: bool,
    output_prefix: str,
) -> Path:
    if is_batch:
        output.mkdir(parents=True, exist_ok=True)
        return output / f"{output_prefix}{input_raster.name}"

    if output.suffix.lower() == ".tif":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    output.mkdir(parents=True, exist_ok=True)
    return output / f"{output_prefix}{input_raster.name}"


def _count_windows(width: int, height: int, window_size: int) -> int:
    return math.ceil(width / window_size) * math.ceil(height / window_size)


def _is_float_dtype(dtype_name: str) -> bool:
    return str(dtype_name).lower().startswith("float")


def _predictor_value(dtype_name: str) -> str:
    return "FLOATING_POINT" if _is_float_dtype(dtype_name) else "2"


def _write_stacked_raster(
    *,
    input_raster: Path,
    output_path: Path,
    output_driver: str,
    bands: list[int],
    window_size: int,
    block_size: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_raster) as src:
        if src.count < max(bands):
            raise ValueError(
                f"Input raster has {src.count} bands but requested bands {bands}."
            )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=len(bands),
            compress="DEFLATE",
            predictor=3 if _is_float_dtype(src.dtypes[0]) else 2,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="IF_SAFER",
        )

        work_output = output_path
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if output_driver == "COG":
            temp_dir = tempfile.TemporaryDirectory(prefix="stacked_bands_")
            work_output = Path(temp_dir.name) / f"{input_raster.stem}_stacked_tmp.tif"

        total_windows = _count_windows(src.width, src.height, window_size)

        with rasterio.open(work_output, "w", **profile) as dst:
            dst.descriptions = tuple(_band_description_from_metadata(src, band) for band in bands)
            dst.update_tags(
                **(src.tags() or {}),
                stacked_source=str(input_raster),
                stacked_source_bands=",".join(str(band) for band in bands),
                stacked_output_driver=output_driver,
            )

            for out_index, in_band in enumerate(bands, start=1):
                try:
                    dst.update_tags(out_index, **(src.tags(in_band) or {}), source_band=in_band)
                except Exception:
                    pass

            progress = tqdm(total=total_windows, desc=f"Stack bands {input_raster.name}", unit="window")
            for y in range(0, src.height, window_size):
                for x in range(0, src.width, window_size):
                    window = Window(
                        col_off=x,
                        row_off=y,
                        width=min(window_size, src.width - x),
                        height=min(window_size, src.height - y),
                    )
                    selected = src.read(bands, window=window)
                    dst.write(selected, window=window)
                    progress.update(1)

            progress.close()

        if output_driver == "COG":
            rio_copy(
                work_output,
                output_path,
                driver="COG",
                COMPRESS="DEFLATE",
                LEVEL="9",
                PREDICTOR=_predictor_value(src.dtypes[0]),
                BLOCKSIZE=str(block_size),
                BIGTIFF="IF_SAFER",
                RESAMPLING="NEAREST",
                OVERVIEWS="AUTO",
                OVERVIEW_COMPRESS="DEFLATE",
                OVERVIEW_PREDICTOR=_predictor_value(src.dtypes[0]),
            )
            temp_dir.cleanup()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    bands = _parse_band_list(args.bands)
    block_size = _normalize_block_size(args.block_size)
    rasters = _iter_input_rasters(input_path)
    is_batch = input_path.is_dir()

    if not rasters:
        raise FileNotFoundError(f"No .tif files found under: {input_path}")

    for raster_path in rasters:
        resolved_output = _resolve_output_path(
            input_raster=raster_path,
            output=output_path,
            is_batch=is_batch,
            output_prefix=args.output_prefix,
        )
        _write_stacked_raster(
            input_raster=raster_path,
            output_path=resolved_output,
            output_driver=args.output_driver,
            bands=bands,
            window_size=int(args.window_size),
            block_size=block_size,
            overwrite=args.overwrite,
        )
        print(f"Wrote stacked predictor: {resolved_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())