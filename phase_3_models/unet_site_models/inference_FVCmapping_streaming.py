#!/usr/bin/env python3
"""Run streaming U-Net inference over a large orthomosaic.

This script is the memory-safe replacement for the legacy
`inference_FVCmapping.py` workflow. It reads one window at a time from a large
input raster, runs inference, and writes the predicted class mask directly to a
GeoTIFF on disk without materializing the full raster in memory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window
from tqdm import tqdm

from phase_3_models.unet_site_models.inference_raster_to_geojson import (
    VARIANT_PRESETS,
    apply_class_id_map,
    load_model,
    parse_class_id_map,
    parse_valid_classes,
    predict_tile,
)


DEFAULT_WINDOW_SIZE = 256
DEFAULT_MASK_NODATA = 255
DEFAULT_TEN_BAND_SELECTION = [2, 4, 6, 8, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run streaming U-Net inference over a full raster and write a class-mask GeoTIFF"
    )

    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_PRESETS.keys()),
        default="medium",
        help="Inference preset controlling default classes/out-channels (default: medium)",
    )
    parser.add_argument("--model-path", required=True, help="Path to a .pth checkpoint")
    parser.add_argument("--input-raster", required=True, help="Path to a large input .tif raster")
    parser.add_argument("--output-mask", required=True, help="Output class-mask GeoTIFF path")
    parser.add_argument(
        "--output-mask-cog",
        default="",
        help="Optional output path for a Cloud-Optimized GeoTIFF copy of the class mask",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Inference window size in pixels (default: 256)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Output raster block size in pixels (default: 256)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for 1-class models")
    parser.add_argument("--positive-class-id", type=int, default=1, help="Positive class id for binary models")
    parser.add_argument(
        "--valid-classes",
        type=str,
        default=None,
        help=(
            "Comma-separated class ids. If omitted, defaults based on --variant "
            "(low: 0,1,2; medium: 0,1,2,3; dense: 0,1,2,3,4)."
        ),
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=5,
        help="Model input channels; if omitted, inferred from checkpoint when possible",
    )
    parser.add_argument(
        "--input-bands",
        type=str,
        default="",
        help=(
            "Optional comma-separated 1-based raster band numbers to feed into the model. "
            "Example: '2,4,6,8,10' for the legacy 10-band MicaSense predictor selection. "
            "If omitted, the script uses bands 1..N for N-band predictors, or defaults to "
            "2,4,6,8,10 when the source raster has 10 bands and the model expects 5 channels."
        ),
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=None,
        help=(
            "Model output channels; if omitted, inferred from checkpoint when possible, "
            "otherwise defaults to number of classes in --variant"
        ),
    )
    parser.add_argument(
        "--class-id-map",
        type=str,
        default="",
        help=(
            "Optional class id remapping applied after prediction. "
            "Format: 'src:dst,src2:dst2'. Example to swap NPV/PV: '1:2,2:1'."
        ),
    )
    parser.add_argument(
        "--mask-nodata",
        type=int,
        default=DEFAULT_MASK_NODATA,
        help="Nodata value for the output class mask (default: 255)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return parser.parse_args()


def _normalize_block_size(value: int) -> int:
    value = max(16, min(512, int(value)))
    value = (value // 16) * 16
    return value if value >= 16 else 16


def _count_windows(width: int, height: int, window_size: int) -> int:
    return math.ceil(width / window_size) * math.ceil(height / window_size)


def _parse_band_selection(raw: str) -> list[int]:
    bands = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not bands:
        raise ValueError("--input-bands must contain at least one band number")
    if min(bands) < 1:
        raise ValueError("--input-bands must use 1-based positive integers")
    return bands


def _resolve_input_bands(*, src_count: int, in_channels: int, raw_input_bands: str) -> list[int]:
    raw_input_bands = str(raw_input_bands or "").strip()
    if raw_input_bands:
        selected_bands = _parse_band_selection(raw_input_bands)
    elif src_count == in_channels:
        selected_bands = list(range(1, in_channels + 1))
    elif src_count == 10 and in_channels == 5:
        selected_bands = list(DEFAULT_TEN_BAND_SELECTION)
    else:
        selected_bands = list(range(1, in_channels + 1))

    if len(selected_bands) != in_channels:
        raise ValueError(
            f"Selected input bands {selected_bands} do not match the model input channel count {in_channels}."
        )
    if max(selected_bands) > src_count:
        raise ValueError(
            f"Selected input bands {selected_bands} exceed the raster band count {src_count}."
        )

    return selected_bands


def _ensure_writable_output(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")


def _build_valid_data_mask(src: rasterio.io.DatasetReader, window: Window, input_bands: list[int]) -> np.ndarray:
    band_masks = src.read_masks(input_bands, window=window)
    valid_data_mask = np.all(band_masks > 0, axis=0)

    source_nodata = src.nodata
    if source_nodata is not None:
        tile = src.read(input_bands, window=window)
        if np.issubdtype(tile.dtype, np.floating) and np.isnan(source_nodata):
            valid_data_mask &= np.all(~np.isnan(tile), axis=0)
        else:
            valid_data_mask &= np.all(tile != source_nodata, axis=0)
        if np.issubdtype(tile.dtype, np.floating):
            valid_data_mask &= np.all(np.isfinite(tile), axis=0)

    return valid_data_mask


def _update_class_counts(class_counts: dict[int, int], array: np.ndarray, *, ignore_value: int | None = None) -> None:
    values, counts = np.unique(array, return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        if ignore_value is not None and int(value) == int(ignore_value):
            continue
        class_counts[int(value)] = class_counts.get(int(value), 0) + int(count)


def _build_mask_cog(source_path: Path, cog_path: Path, block_size: int, overwrite: bool) -> None:
    _ensure_writable_output(cog_path, overwrite=overwrite)
    rio_copy(
        source_path,
        cog_path,
        driver="COG",
        COMPRESS="DEFLATE",
        LEVEL="9",
        PREDICTOR="2",
        BLOCKSIZE=str(block_size),
        BIGTIFF="IF_SAFER",
        RESAMPLING="NEAREST",
        OVERVIEWS="AUTO",
        OVERVIEW_COMPRESS="DEFLATE",
        OVERVIEW_PREDICTOR="2",
    )


def run_streaming_inference(
    *,
    model,
    input_raster: Path,
    output_mask: Path,
    window_size: int,
    block_size: int,
    in_channels: int,
    input_bands: list[int],
    valid_classes: list[int],
    class_id_map: dict[int, int],
    mask_nodata: int,
) -> tuple[dict[int, int], list[int]]:
    class_counts: dict[int, int] = {}
    invalid_classes: set[int] = set()

    with rasterio.open(input_raster) as src:
        if src.count < in_channels:
            raise ValueError(
                f"Input raster has {src.count} bands but the model expects {in_channels} channels."
            )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype=rasterio.uint8,
            nodata=int(mask_nodata),
            compress="DEFLATE",
            predictor=2,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="IF_SAFER",
        )

        output_mask.parent.mkdir(parents=True, exist_ok=True)
        total_windows = _count_windows(src.width, src.height, window_size)

        with rasterio.open(output_mask, "w", **profile) as dst:
            dst.update_tags(
                source_raster=str(input_raster),
                source_width=int(src.width),
                source_height=int(src.height),
                inference_window_size=int(window_size),
                inference_in_channels=int(in_channels),
                inference_input_bands=",".join(str(v) for v in input_bands),
                inference_valid_classes=",".join(str(v) for v in valid_classes),
                class_id_map=",".join(f"{k}:{v}" for k, v in sorted(class_id_map.items())),
            )

            progress = tqdm(total=total_windows, desc="Streaming inference", unit="window")
            for y in range(0, src.height, window_size):
                for x in range(0, src.width, window_size):
                    window = Window(
                        col_off=x,
                        row_off=y,
                        width=min(window_size, src.width - x),
                        height=min(window_size, src.height - y),
                    )
                    valid_data_mask = _build_valid_data_mask(src, window, input_bands)

                    if not np.any(valid_data_mask):
                        pred = np.full(
                            (int(window.height), int(window.width)),
                            fill_value=int(mask_nodata),
                            dtype=np.uint8,
                        )
                        dst.write(pred, 1, window=window)
                        progress.update(1)
                        continue

                    tile = src.read(input_bands, window=window)
                    if not np.all(valid_data_mask):
                        tile = tile.astype(np.float32, copy=False)
                        tile[:, ~valid_data_mask] = 0

                    pred = predict_tile(model, tile, in_channels=in_channels)

                    if class_id_map:
                        pred = apply_class_id_map(pred, class_id_map)

                    pred = pred.astype(np.uint8, copy=False)
                    pred[~valid_data_mask] = np.uint8(mask_nodata)
                    dst.write(pred, 1, window=window)
                    _update_class_counts(class_counts, pred, ignore_value=int(mask_nodata))

                    for value in np.unique(pred).tolist():
                        value = int(value)
                        if value == int(mask_nodata):
                            continue
                        if value not in valid_classes:
                            invalid_classes.add(value)

                    progress.update(1)

            progress.close()

    return class_counts, sorted(invalid_classes)


def main() -> int:
    args = parse_args()

    input_raster = Path(args.input_raster)
    output_mask = Path(args.output_mask)
    output_mask_cog = Path(args.output_mask_cog) if str(args.output_mask_cog).strip() else None

    if not input_raster.exists():
        raise FileNotFoundError(f"Input raster not found: {input_raster}")

    _ensure_writable_output(output_mask, overwrite=args.overwrite)

    if output_mask_cog is not None and output_mask_cog.resolve() == output_mask.resolve():
        raise ValueError("--output-mask-cog must be different from --output-mask")

    predict_tile.threshold = args.threshold
    predict_tile.positive_class_id = args.positive_class_id

    preset = VARIANT_PRESETS[args.variant]
    preset_valid = ",".join(str(v) for v in sorted(preset["class_labels"].values()))
    valid_classes = parse_valid_classes(args.valid_classes if args.valid_classes is not None else preset_valid)
    out_channels = int(args.out_channels) if args.out_channels is not None else len(valid_classes)
    class_id_map = parse_class_id_map(args.class_id_map)

    model, resolved_in_channels, resolved_out_channels = load_model(
        args.model_path,
        in_channels=int(args.in_channels),
        out_channels=out_channels,
    )

    block_size = _normalize_block_size(args.block_size)
    with rasterio.open(input_raster) as src:
        input_bands = _resolve_input_bands(
            src_count=int(src.count),
            in_channels=resolved_in_channels,
            raw_input_bands=args.input_bands,
        )

    class_counts, invalid_classes = run_streaming_inference(
        model=model,
        input_raster=input_raster,
        output_mask=output_mask,
        window_size=int(args.window_size),
        block_size=block_size,
        in_channels=resolved_in_channels,
        input_bands=input_bands,
        valid_classes=valid_classes,
        class_id_map=class_id_map,
        mask_nodata=int(args.mask_nodata),
    )

    print(f"Wrote mask GeoTIFF: {output_mask}")
    print(f"Resolved channels -> in: {resolved_in_channels}, out: {resolved_out_channels}")
    print(f"Input bands used: {input_bands}")
    print(f"Mask classes (value:count): {class_counts}")
    if invalid_classes:
        print(f"Warning: mask contains classes outside --valid-classes {valid_classes}: {invalid_classes}")

    if output_mask_cog is not None:
        _build_mask_cog(output_mask, output_mask_cog, block_size=block_size, overwrite=args.overwrite)
        print(f"Wrote mask COG: {output_mask_cog}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())