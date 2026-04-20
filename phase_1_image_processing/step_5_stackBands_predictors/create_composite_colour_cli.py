#!/usr/bin/env python3
"""Create a georeferenced 3-band color composite from a multispectral raster.

This is the parameterized replacement for `5_1_create_compositeColour.py`.
It estimates stretch values from a downsampled preview, then streams the full
input raster window-by-window to avoid loading large orthomosaics into memory.

The output can be written either as a compressed GeoTIFF or a Cloud-Optimized
GeoTIFF suitable for STAC asset publication.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window
from tqdm import tqdm


DEFAULT_OUTPUT_PREFIX = "composite_percentile_"
DEFAULT_SAMPLE_SIZE = 2048
DEFAULT_WINDOW_SIZE = 1024
DEFAULT_OUTPUT_NODATA = 0
DEFAULT_INPUT_NODATA = -32767.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a parameterized color composite from one raster or a directory of rasters"
    )
    parser.add_argument("--input", required=True, help="Input .tif file or a directory containing .tif rasters")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .tif path (single input) or output directory (directory input)",
    )
    parser.add_argument(
        "--bands",
        default="10,6,2",
        help="Comma-separated 1-based source bands to map to R,G,B (default: 10,6,2)",
    )
    parser.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for stretch (default: 1)")
    parser.add_argument("--pmax", type=float, default=99.0, help="Upper percentile for stretch (default: 99)")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Longest preview dimension used for percentile estimation (default: 2048)",
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
        "--input-nodata",
        type=float,
        default=DEFAULT_INPUT_NODATA,
        help="Fallback nodata value when the source raster does not define one (default: -32767)",
    )
    parser.add_argument(
        "--output-nodata",
        type=int,
        default=DEFAULT_OUTPUT_NODATA,
        help="Output nodata value for the uint8 composite (default: 0)",
    )
    parser.add_argument(
        "--output-driver",
        choices=["GTiff", "COG"],
        default="COG",
        help="Output driver (default: COG)",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for output files when --input is a directory (default: composite_percentile_)",
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
    if len(bands) != 3:
        raise ValueError("--bands must specify exactly 3 band numbers, e.g. '10,6,2'")
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


def _estimate_percentiles(
    src: rasterio.io.DatasetReader,
    bands: list[int],
    *,
    pmin: float,
    pmax: float,
    sample_size: int,
    nodata_value: float | None,
) -> list[tuple[float, float]]:
    scale = min(1.0, sample_size / max(src.width, src.height))
    out_width = max(1, int(math.ceil(src.width * scale)))
    out_height = max(1, int(math.ceil(src.height * scale)))

    stats: list[tuple[float, float]] = []
    for band_index in bands:
        preview = src.read(
            band_index,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        if nodata_value is not None:
            preview = np.where(preview == nodata_value, np.nan, preview)

        finite = preview[np.isfinite(preview)]
        if finite.size == 0:
            stats.append((float("nan"), float("nan")))
            continue

        lo = float(np.nanpercentile(finite, pmin))
        hi = float(np.nanpercentile(finite, pmax))
        stats.append((lo, hi))

    return stats


def _stretch_to_byte(
    array: np.ndarray,
    *,
    lo: float,
    hi: float,
    nodata_value: float | None,
    output_nodata: int,
) -> np.ndarray:
    working = array.astype(np.float32, copy=True)
    invalid_mask = ~np.isfinite(working)
    if nodata_value is not None:
        invalid_mask |= working == nodata_value
    working[invalid_mask] = np.nan

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(working.shape, output_nodata, dtype=np.uint8)

    scaled = np.clip((working - lo) / (hi - lo) * 254.0, 0.0, 254.0) + 1.0
    scaled = np.where(np.isnan(scaled), output_nodata, scaled)
    return scaled.astype(np.uint8)


def _count_windows(width: int, height: int, window_size: int) -> int:
    return math.ceil(width / window_size) * math.ceil(height / window_size)


def _copy_to_cog(source_path: Path, output_path: Path, block_size: int, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rio_copy(
        source_path,
        output_path,
        driver="COG",
        COMPRESS="DEFLATE",
        LEVEL="9",
        PREDICTOR="2",
        BLOCKSIZE=str(block_size),
        BIGTIFF="IF_SAFER",
        RESAMPLING="BILINEAR",
        OVERVIEWS="AUTO",
        OVERVIEW_COMPRESS="DEFLATE",
        OVERVIEW_PREDICTOR="2",
    )


def _write_composite(
    *,
    input_raster: Path,
    output_path: Path,
    output_driver: str,
    bands: list[int],
    band_stats: list[tuple[float, float]],
    window_size: int,
    block_size: int,
    output_nodata: int,
    input_nodata: float | None,
    pmin: float,
    pmax: float,
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
            dtype=rasterio.uint8,
            count=3,
            nodata=int(output_nodata),
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            photometric="RGB",
            BIGTIFF="IF_SAFER",
        )

        work_output = output_path
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if output_driver == "COG":
            temp_dir = tempfile.TemporaryDirectory(prefix="composite_colour_")
            work_output = Path(temp_dir.name) / f"{input_raster.stem}_composite_tmp.tif"

        total_windows = _count_windows(src.width, src.height, window_size)

        with rasterio.open(work_output, "w", **profile) as dst:
            dst.descriptions = tuple(_band_description_from_metadata(src, band) for band in bands)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.update_tags(
                **(src.tags() or {}),
                composite_source=str(input_raster),
                composite_source_bands=",".join(str(band) for band in bands),
                composite_pmin=str(pmin),
                composite_pmax=str(pmax),
                composite_output_driver=output_driver,
            )

            for out_index, in_band in enumerate(bands, start=1):
                try:
                    dst.update_tags(
                        out_index,
                        **(src.tags(in_band) or {}),
                        source_band=in_band,
                        stretch_min=band_stats[out_index - 1][0],
                        stretch_max=band_stats[out_index - 1][1],
                    )
                except Exception:
                    pass

            progress = tqdm(total=total_windows, desc=f"Composite {input_raster.name}", unit="window")
            for y in range(0, src.height, window_size):
                for x in range(0, src.width, window_size):
                    window = Window(
                        col_off=x,
                        row_off=y,
                        width=min(window_size, src.width - x),
                        height=min(window_size, src.height - y),
                    )

                    rgb = []
                    for band_index, (lo, hi) in zip(bands, band_stats):
                        band = src.read(band_index, window=window)
                        rgb.append(
                            _stretch_to_byte(
                                band,
                                lo=lo,
                                hi=hi,
                                nodata_value=input_nodata,
                                output_nodata=output_nodata,
                            )
                        )

                    dst.write(np.stack(rgb, axis=0), window=window)
                    progress.update(1)

            progress.close()

        if output_driver == "COG":
            _copy_to_cog(work_output, output_path, block_size=block_size, overwrite=overwrite)
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

        with rasterio.open(raster_path) as src:
            nodata_value = src.nodata if src.nodata is not None else args.input_nodata
            band_stats = _estimate_percentiles(
                src,
                bands,
                pmin=float(args.pmin),
                pmax=float(args.pmax),
                sample_size=int(args.sample_size),
                nodata_value=nodata_value,
            )

        _write_composite(
            input_raster=raster_path,
            output_path=resolved_output,
            output_driver=args.output_driver,
            bands=bands,
            band_stats=band_stats,
            window_size=int(args.window_size),
            block_size=block_size,
            output_nodata=int(args.output_nodata),
            input_nodata=nodata_value,
            pmin=float(args.pmin),
            pmax=float(args.pmax),
            overwrite=args.overwrite,
        )
        print(f"Wrote composite: {resolved_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())